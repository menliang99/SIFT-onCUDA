// file:        sift.cu
// author:      Liang Men
// email: mliang@uark.edu


#include<sift.hpp>
#include<sift-conv.tpp>

#include<algorithm>
#include<iostream>
#include<sstream>
#include<cassert>



#include <cutil.h>


extern "C" {
#if defined (VL_MAC)
#include<libgen.h>
#else
#include<string.h>
}
#endif

#define BLOCK_SIZE 16
#define C_TILE_SIZE 250
#define D_BLOCK_SIZE 128

#define F_TILE_SIZE 14
#define F_BLOCK_SIZE F_TILE_SIZE+2

#define F_TILE_SIZE_S 12
#define F_BLOCK_SIZE_S F_TILE_SIZE_S+4

#define G_TILE_SIZE 14
#define G_BLOCK_SIZE G_TILE_SIZE+2

#define K_BLOCK_SIZE 128

using namespace VL ;

// on startup, pre-compute expn(x) = exp(-x)
namespace VL { 
namespace Detail {

int const         expnTableSize = 256 ;
VL::float_t const expnTableMax  = VL::float_t(25.0) ;
VL::float_t       expnTable [ expnTableSize + 1 ] ;

struct buildExpnTable
{
  buildExpnTable() {
    for(int k = 0 ; k < expnTableSize + 1 ; ++k) {
      expnTable[k] = exp( - VL::float_t(k) / expnTableSize * expnTableMax ) ;
    }
  }
} _buildExpnTable ;

} }


namespace VL {
namespace Detail {

/** Comment eater istream manipulator */
class _cmnt {} cmnt ;

/** @brief Extract a comment from a stream
 **
 ** The function extracts a block of consecutive comments from an
 ** input stream. A comment is a sequence of whitespaces, followed by
 ** a `#' character, other characters and terminated at the next line
 ** ending. A block of comments is just a sequence of comments.
 **/
std::istream& 
operator>>(std::istream& is, _cmnt& manip)
{
  char c ;
  char b [1024] ; 
  is>>c ;
  if( c != '#' ) 
    return is.putback(c) ;
  is.getline(b,1024) ;
  return is ;
}

}

/** @brief Insert PGM file into stream
 **
 ** The function iserts into the stream @a os the grayscale image @a
 ** im encoded as a PGM file. The immage is assumed to be normalized
 ** in the range 0.0 - 1.0.
 **
 ** @param os output stream.
 ** @param im pointer to image data.
 ** @param width image width.
 ** @param height image height.
 ** @return the stream @a os.
 **/
std::ostream& 
insertPgm(std::ostream& os, pixel_t const* im, int width, int height)
{
  os<< "P5"   << "\n"
    << width  << " "
    << height << "\n"
    << "255"  << "\n" ;
  for(int y = 0 ; y < height ; ++y) {
    for(int x = 0 ; x < width ; ++x) {
      unsigned char v = 
        (unsigned char)
        (std::max(std::min(*im++, 1.0f),0.f) * 255.0f) ;
      os << v ;
    }
  }
  return os ;
}

/** @brief Extract PGM file from stream.
 **
 ** The function extracts from the stream @a in a grayscale image
 ** encoded as a PGM file. The function fills the structure @a buffer,
 ** containing the image dimensions and a pointer to the image data.
 **
 ** The image data is an array of floats and is owned by the caller,
 ** which should erase it as in
 ** 
 ** @code
 **   delete [] buffer.data.
 ** @endcode
 **
 ** When the function encouters an error it throws a generic instance
 ** of VL::Exception.
 **
 ** @param in input stream.
 ** @param buffer buffer descriptor to be filled.
 ** @return the stream @a in.
 **/
std::istream& 
extractPgm(std::istream& in, PgmBuffer& buffer)
{
  pixel_t* im_pt ;
  int      width ;
  int      height ;
  int      maxval ;

  char c ;
  in>>c ;
  if( c != 'P') VL_THROW("File is not in PGM format") ;
  
  bool is_ascii ;
  in>>c ;
  switch( c ) {
  case '2' : is_ascii = true ; break ;
  case '5' : is_ascii = false ; break ;
  default  : VL_THROW("File is not in PGM format") ;
  }
  
  in >> Detail::cmnt
     >> width
     >> Detail::cmnt 
     >> height
     >> Detail::cmnt
     >> maxval ;

  // after maxval no more comments, just a whitespace or newline
  {char trash ; in.get(trash) ;}

  if(maxval > 255)
    VL_THROW("Only <= 8-bit per channel PGM files are supported") ;

  if(! in.good()) 
    VL_THROW("PGM header parsing error") ;
  
  im_pt = new pixel_t [ width*height ];
  
  try {
    if( is_ascii ) {
      pixel_t* start = im_pt ;
      pixel_t* end   = start + width*height ; 
      pixel_t  norm  = pixel_t( maxval ) ;
      
      while( start != end ) {        
        int i ;
        in >> i ;	
        if( ! in.good() ) VL_THROW
                            ("PGM parsing error file (width="<<width
                             <<" height="<<height
                             <<" maxval="<<maxval
                             <<" at pixel="<<start-im_pt<<")") ;    
        *start++ = pixel_t( i ) / norm ;        
      }
    } else {
      std::streampos beg = in.tellg() ;
      char* buffer = new char [width*height] ;
      in.read(buffer, width*height) ;
      if( ! in.good() ) VL_THROW
			  ("PGM parsing error file (width="<<width
			   <<" height="<<height
			   <<" maxval="<<maxval
			   <<" at pixel="<<in.tellg()-beg<<")") ;
      
      pixel_t* start = im_pt ;
      pixel_t* end   = start + width*height ; 
      uint8_t* src = reinterpret_cast<uint8_t*>(buffer) ;      
      while( start != end ) *start++ = *src++ / 255.0f ;
    }       
  } catch(...) {
    delete [] im_pt ; 
    throw ;
  }
  
  buffer.width  = width ;
  buffer.height = height ;
  buffer.data   = im_pt ;

  return in ;
}

// ===================================================================
//                                          Low level image operations
// -------------------------------------------------------------------

namespace Detail {

/** @brief Copy an image
 ** @param dst    output imgage buffer.
 ** @param src    input image buffer.
 ** @param width  input image width.
 ** @param height input image height.
 **/
void
copy(pixel_t* dst, pixel_t const* src, int width, int height)
{
  memcpy(dst, src, sizeof(pixel_t)*width*height)  ;
}

/** @brief Copy an image upsampling two times
 **
 ** The destination buffer must be at least as big as two times the
 ** input buffer. Bilinear interpolation is used.
 **
 ** @param dst     output imgage buffer.
 ** @param src     input image buffer.
 ** @param width   input image width.
 ** @param height  input image height.
 **/

/*
void 
copyAndUpsampleRows
(pixel_t* dst, pixel_t const* src, int width, int height)
{
  for(int y = 0 ; y < height ; ++y) {
    pixel_t b, a ;
    b = a = *src++ ;
    for(int x = 0 ; x < width-1 ; ++x) {
      b = *src++ ;
      *dst = a ;         dst += height ;
      *dst = 0.5*(a+b) ; dst += height ;
      a = b ;
    }
    *dst = b ; dst += height ;
    *dst = b ; dst += height ;
    dst += 1 - width * 2 * height ;
  }  
}


*/
void 
copyAndDownsample(pixel_t* dst, pixel_t const* src, 
                  int width, int height, int d)
{
  for(int y = 0 ; y < height ; y+=d) {
    pixel_t const * srcrowp = src + y * width ;    
    for(int x = 0 ; x < width - (d-1) ; x+=d) {     
      *dst++ = *srcrowp ;
      srcrowp += d ;
    }
  }
}


}

__global__ void UpsampleKernel(pixel_t* dst, pixel_t* src, int src_width, int src_height)
{

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Col = bx*BLOCK_SIZE + tx;
  int Row = by*BLOCK_SIZE + ty;

  if ( Col < (src_width -1) && Row < src_height )
  {
  dst[2*Col*src_height + Row] = src[Row*src_width + Col];
  dst[(2*Col+1)*src_height + Row] = 
         (src[Row*src_width + Col] + src[Row*src_width + Col + 1])/2;
  }
  else
  {
    if ( Col == (src_width - 1) && Row < src_height )
     {
      dst[2*Col*src_height + Row] = src[Row*src_width + Col];
      dst[(2*Col+1)*src_height + Row] = src[Row*src_width + Col];
      }
   }
}


void 
copyAndUpsampleRows (pixel_t* dst, pixel_t const* src, int width, int height)

{
    int dst_width = height;
    int dst_height = width * 2;


    unsigned int src_size = sizeof(pixel_t) * (width*height);
    unsigned int dst_size = sizeof(pixel_t) * (dst_width*dst_height);

    pixel_t* dst_d = NULL;
    pixel_t* src_d = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &src_d, src_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &dst_d, dst_size));

    CUDA_SAFE_CALL( cudaMemcpy( src_d, src, src_size, cudaMemcpyHostToDevice) );

    dim3 dimBlock, dimGrid1;
    dimBlock.x = dimBlock.y = BLOCK_SIZE;
    dimBlock.z = 1;
    dimGrid1.x = (width / dimBlock.x) + ( (width % dimBlock.x) ? 1:0 );
    dimGrid1.y = (height / dimBlock.y) + ( (height % dimBlock.y) ? 1:0 );
    dimGrid1.z = 1;

    UpsampleKernel<<<dimGrid1, dimBlock>>>(dst_d, src_d, width, height);
    
    CUDA_SAFE_CALL(cudaMemcpy( dst, dst_d, dst_size, cudaMemcpyDeviceToHost));
    cudaFree(dst_d);
    cudaFree(src_d);

}


void   //Use this function to reduce double call in the main function.
copyAndUpsampleRows2 (pixel_t* dst, pixel_t const* src, int width, int height)

{
    int tmp_width = height;
    int tmp_height = width * 2;
    int dst_width = width * 2;
    int dst_height = height * 2;

    unsigned int src_size = sizeof(pixel_t) * (width*height);
    unsigned int tmp_size = sizeof(pixel_t) * (2*width*height);
    unsigned int dst_size = sizeof(pixel_t) * (dst_width*dst_height);

    pixel_t* dst_d = NULL;
    pixel_t* tmp_d = NULL;
    pixel_t* src_d = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &src_d, src_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &tmp_d, tmp_size));

    CUDA_SAFE_CALL( cudaMemcpy( src_d, src, src_size, cudaMemcpyHostToDevice) );

    dim3 dimBlock, dimGrid1, dimGrid2;
    dimBlock.x = dimBlock.y = BLOCK_SIZE;
    dimBlock.z = 1;
    dimGrid1.x = (width / dimBlock.x) + ( (width % dimBlock.x) ? 1:0 );
    dimGrid1.y = (height / dimBlock.y) + ( (height % dimBlock.y) ? 1:0 );
    dimGrid1.z = 1;
    dimGrid2.x = (tmp_width / dimBlock.x) + ( (tmp_width % dimBlock.x) ? 1:0 );
    dimGrid2.y = (tmp_height / dimBlock.y) + ( (tmp_height % dimBlock.y) ? 1:0 );
    dimGrid2.z = 1;

    UpsampleKernel<<<dimGrid1, dimBlock>>>(tmp_d, src_d, width, height);

    cudaFree(src_d);
    CUDA_SAFE_CALL( cudaMalloc( (void**) &dst_d, dst_size));

    UpsampleKernel<<<dimGrid2, dimBlock>>>(dst_d, tmp_d, width, height);

    CUDA_SAFE_CALL(cudaMemcpy( dst, dst_d, dst_size, cudaMemcpyDeviceToHost));
    cudaFree(dst_d);
    cudaFree(tmp_d);

}

/*

__global__ void DownsampleKernel(pixel_t* dst, pixel_t* src, int src_width, int src_height, int dst_width,  int d)
{



  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;



  int Col = bx*BLOCK_SIZE + tx;
  int Row = by*BLOCK_SIZE + ty;


  if ( d*Col < src_width && d*Row < src_height)
  dst[Row*dst_width + Col] = src[d*Row*src_width + d*Col];

}

void copyAndDownsample(pixel_t* dst, pixel_t const* src, int width, int height, int d)
{

    int dst_width = (width / d) + ((width % d) ? 1:0 );
    int dst_height =(height / d) + ((height % d) ? 1:0);

    unsigned int src_size = sizeof(pixel_t) * (width*height);
    unsigned int dst_size = sizeof(pixel_t) * (dst_width*dst_height);

    pixel_t* dst_d = NULL;
    pixel_t* src_d = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &dst_d, dst_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &src_d, src_size));

    CUDA_SAFE_CALL( cudaMemcpy( src_d, src, src_size, cudaMemcpyHostToDevice) );

    dim3 dimBlock, dimGrid;
    dimBlock.x = dimBlock.y = BLOCK_SIZE;
    dimBlock.z = 1;
    dimGrid.x = (dst_width / dimBlock.x) + ( (dst_width % dimBlock.x) ? 1:0 );
    dimGrid.y = (dst_height / dimBlock.y) + ( (dst_height % dimBlock.y) ? 1:0 );
    dimGrid.z = 1;

    DownsampleKernel<<<dimGrid, dimBlock>>>(dst_d, src_d, width, height, dst_width, d);

    CUDA_SAFE_CALL(cudaMemcpy( dst, dst_d, dst_size, cudaMemcpyDeviceToHost));
    cudaFree(dst_d);
    cudaFree(src_d);
}

*/
/*
void econvolve(pixel_t*       dst_pt, 
	   pixel_t* src_pt,    int M, int N,
	   pixel_t* filter_pt, int W)
{
  //typedef T const TC ;
  // convolve along columns, save transpose
  // image is M by N 
  // buffer is N by M 
  // filter is (2*W+1) by 1





  for(int j = 0 ; j < N ; ++j) {



    for(int i = 0 ; i < M ; ++i) {
      pixel_t   acc = 0.0 ;
      pixel_t* g = filter_pt ;
      pixel_t* start = src_pt + (i-W) ;
      pixel_t* stop  ;
      pixel_t   x ;

      // beginning
      stop = src_pt + std::max(0, i-W) ;
      x    = *stop ;
      while( start <= stop ) { acc += (*g++) * x ; start++ ; }

      // middle
      stop =  src_pt + std::min(M-1, i+W) ;
      while( start <  stop ) acc += (*g++) * (*start++) ;

      // end
      x  = *start ;
      stop = src_pt + (i+W) ;
      while( start <= stop ) { acc += (*g++) * x ; start++ ; } 
   
      // save 
      *dst_pt = acc ; 
      dst_pt += N ;

      assert( g - filter_pt == 2*W+1 ) ;

    }
    // next column
    src_pt += M ;
    dst_pt -= M*N - 1 ;


  }
}
*/



__global__ void ConvKernel(pixel_t* dst, pixel_t* src, int src_width, int src_height, pixel_t* filter, int w)

{
 extern __shared__ pixel_t Ns[];  

int tx = threadIdx.x;
int bx = blockIdx.x;
int by = blockIdx.y;

int Row = by;
int Col = bx*C_TILE_SIZE + tx;

int i;

pixel_t Pvalue = 0;

 if ((Col - w) >= 0 && (Col - w) <= (src_width - 1))
   {
    Ns[tx] = src[Row * src_width + (Col - w)];
   }
 else
   {
	if((Col - w) < 0)          
    	   Ns[tx] = src[Row * src_width];  
        else
           Ns[tx] = src[(Row + 1) * src_width - 1];
    }
   
__syncthreads();

 if (tx < C_TILE_SIZE)
  {

   for ( i = 0; i < 2*w+1; i++)    
     Pvalue += filter[i] * Ns[i+tx];
  
   if (Col < src_width )
     dst[Col * src_height + Row] = Pvalue;

  }
}

void  econvolve(pixel_t* dst, pixel_t* src, int src_width, int src_height,
	  pixel_t* filter_pt, int W)
{

  // convolve along columns, save transpose
  // image is M by N 
  // buffer is N by M 
  // filter is (2*W+1) by 1

    unsigned int src_size = sizeof(pixel_t) * (src_width*src_height);
    unsigned int dst_size = sizeof(pixel_t) * (src_width*src_height);
    unsigned int filter_size = sizeof(pixel_t) * (2*W + 1);

    pixel_t* dst_d = NULL;
    pixel_t* src_d = NULL;
    pixel_t* filter_d = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &dst_d, dst_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &src_d, src_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &filter_d, filter_size));

    CUDA_SAFE_CALL( cudaMemcpy( src_d, src, src_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( filter_d, filter_pt, filter_size, cudaMemcpyHostToDevice) );
 

    int SizeofSM = sizeof(pixel_t) * (2*W + C_TILE_SIZE);
    dim3 dimBlock, dimGrid;
    dimBlock.x = 2*W + C_TILE_SIZE;
    dimBlock.y = 1;   
    dimGrid.x = (src_width / C_TILE_SIZE) + ( (src_width % C_TILE_SIZE) ? 1:0 );
    dimGrid.y = src_height;

    //  std::cout
    //    << "econvolve:   number of w     : " << W<<std::endl;


    ConvKernel<<<dimGrid, dimBlock, SizeofSM>>>(dst_d, src_d, src_width, src_height, filter_d, W);

    CUDA_SAFE_CALL(cudaMemcpy( dst, dst_d, dst_size, cudaMemcpyDeviceToHost));
    cudaFree(dst_d);
    cudaFree(src_d);
    cudaFree(filter_d);


} 
  


/** @brief Smooth an image 
 **
 ** The function convolves the image @a src by a Gaussian kernel of
 ** variance @a s and writes the result to @a dst. The function also
 ** needs a scratch buffer @a dst of the same size of @a src and @a
 ** dst.
 **
 ** @param dst output image buffer.
 ** @param temp scratch image buffer.
 ** @param src input image buffer.
 ** @param width width of the buffers.
 ** @param height height of the buffers.
 ** @param s standard deviation of the Gaussian kernel.
 **/
void
Sift::smooth
(pixel_t* dst, pixel_t* temp, 
 pixel_t* src, int width, int height, 
 VL::float_t s)
{
  // make sure a buffer larege enough has been allocated
  // to hold the filter
  int W = int( ceil( VL::float_t(4.0) * s ) ) ;


  if( ! filter ) {
    filterReserved = 0 ;
  }
  
  if( filterReserved < W ) {
    filterReserved = W ;
    if( filter ) delete [] filter ;
    filter = new pixel_t [ 2* filterReserved + 1 ] ;
  }
  
  // pre-compute filter
  for(int j = 0 ; j < 2*W+1 ; ++j) 
    filter[j] = VL::pixel_t
      (std::exp
       (VL::float_t
        (-0.5 * (j-W) * (j-W) / (s*s) ))) ;
  
  // normalize to one
  normalize(filter, W) ;
  
  // convolve
  econvolve(temp, src, width, height, filter, W) ;
  econvolve(dst, temp, height, width, filter, W) ;
}

// ===================================================================
//                                                     Sift(), ~Sift()
// -------------------------------------------------------------------

/** @brief Initialize Gaussian scale space parameters
 **
 ** @param _im_pt  Source image data
 ** @param _width  Soruce image width
 ** @param _height Soruce image height
 ** @param _sigman Nominal smoothing value of the input image.
 ** @param _sigma0 Base smoothing level.
 ** @param _O      Number of octaves.
 ** @param _S      Number of levels per octave.
 ** @param _omin   First octave.
 ** @param _smin   First level in each octave.
 ** @param _smax   Last level in each octave.
 **/
Sift::Sift(const pixel_t* _im_pt, int _width, int _height,
     VL::float_t _sigman,
     VL::float_t _sigma0,
     int _O, int _S,
     int _omin, int _smin, int _smax)
  : sigman( _sigman ), 
    sigma0( _sigma0 ),
    O( _O ),
    S( _S ),
    omin( _omin ),
    smin( _smin ),
    smax( _smax ),

    magnif( 3.0f ),
    normalizeDescriptor( true ),
    
    temp( NULL ),
    octaves( NULL ),
    filter( NULL )    
{
  process(_im_pt, _width, _height) ;
}

/** @brief Destroy SIFT filter.
 **/
Sift::~Sift()
{
  freeBuffers() ;
}

/** Allocate buffers. Buffer sizes depend on the image size and the
 ** value of omin.
 **/
void
Sift::
prepareBuffers()
{
  // compute buffer size
  int w = (omin >= 0) ? (width  >> omin) : (width  << -omin) ;
  int h = (omin >= 0) ? (height >> omin) : (height << -omin) ;
  int size = w*h* std::max
    ((smax - smin), 2*((smax+1) - (smin-2) +1)) ;

  if( temp && tempReserved == size ) return ;
  
  freeBuffers() ;
  
  // allocate
  Kmid		 = new kvalue [w*h];
  KeyNum	 = new int [O-omin];
  temp           = new pixel_t [ size ] ; 
  tempReserved   = size ;
  tempIsGrad     = false ;
  tempOctave     = 0 ;

  octaves = new pixel_t* [ O ] ;
  for(int o = 0 ; o < O ; ++o) {
    octaves[o] = new pixel_t [ (smax - smin + 1) * w * h ] ;
    w >>= 1 ;
    h >>= 1 ;
  }
}
  
/** @brief Free buffers.
 **
 ** This function releases any buffer allocated by prepareBuffers().
 **
 ** @sa prepareBuffers().
 **/
void
Sift::
freeBuffers()
{
  if( filter ) {
    delete [] filter ;
  }
  filter = 0 ;

  if( octaves ) {
    for(int o = 0 ; o < O ; ++o) {
      delete [] octaves[ o ] ;
    }
    delete [] octaves ;
  }
  octaves = 0 ;
  
  if( temp ) {
    delete [] temp ;   
  }
  temp = 0  ; 
}

// ===================================================================
//                                                         getKeypoint
// -------------------------------------------------------------------

/** @brief Get keypoint from position and scale
 **
 ** The function returns a keypoint with a given position and
 ** scale. Note that the keypoint structure contains fields that make
 ** sense only in conjunction with a specific scale space. Therefore
 ** the keypoint structure should be re-calculated whenever the filter
 ** is applied to a new image, even if the parameters @a x, @a y and
 ** @a sigma do not change.
 **
 ** @param x x coordinate of the center.
 ** @peram y y coordinate of the center.
 ** @param sigma scale.
 ** @return Corresponing keypoint.
 **/
Sift::Keypoint
Sift::getKeypoint(VL::float_t x, VL::float_t y, VL::float_t sigma) const
{

  /*
    The formula linking the keypoint scale sigma to the octave and
    scale index is

    (1) sigma(o,s) = sigma0 2^(o+s/S)

    for which
    
    (2) o + s/S = log2 sigma/sigma0 == phi.

    In addition to the scale index s (which can be fractional due to
    scale interpolation) a keypoint has an integer scale index is too
    (which is the index of the scale level where it was detected in
    the DoG scale space). We have the constraints:
 
    - o and is are integer

    - is is in the range [smin+1, smax-2  ]

    - o  is in the range [omin,   omin+O-1]

    - is = rand(s) most of the times (but not always, due to the way s
      is obtained by quadratic interpolation of the DoG scale space).

    Depending on the values of smin and smax, often (2) has multiple
    solutions is,o that satisfy all constraints.  In this case we
    choose the one with biggest index o (this saves a bit of
    computation).

    DETERMINING THE OCTAVE INDEX O

    From (2) we have o = phi - s/S and we want to pick the biggest
    possible index o in the feasible range. This corresponds to
    selecting the smallest possible index s. We write s = is + ds
    where in most cases |ds|<.5 (but in general |ds|<1). So we have

       o = phi - s/S,   s = is + ds ,   |ds| < .5 (or |ds| < 1).

    Since is is in the range [smin+1,smax-2], s is in the range
    [smin+.5,smax-1.5] (or [smin,smax-1]), the number o is an integer
    in the range phi+[-smax+1.5,-smin-.5] (or
    phi+[-smax+1,-smin]). Thus the maximum value of o is obtained for
    o = floor(phi-smin-.5) (or o = floor(phi-smin)).

    Finally o is clamped to make sure it is contained in the feasible
    range.

    DETERMINING THE SCALE INDEXES S AND IS

    Given o we can derive is by writing (2) as

      s = is + ds = S(phi - o).

    We then take is = round(s) and clamp its value to be in the
    feasible range.
  */

  int o,ix,iy,is ;
  VL::float_t s,phi ;

  phi = log2(sigma/sigma0) ;
  o   = fast_floor( phi -  (VL::float_t(smin)+.5)/S ) ;
  o   = std::min(o, omin+O-1) ;
  o   = std::max(o, omin    ) ;
  s   = S * (phi - o) ;

  is  = int(s + 0.5) ;
  is  = std::min(is, smax - 2) ;
  is  = std::max(is, smin + 1) ;
  
  VL::float_t per = getOctaveSamplingPeriod(o) ;
  ix = int(x / per + 0.5) ;
  iy = int(y / per + 0.5) ;
  
  Keypoint key ;
  key.o  = o ;

  key.ix = ix ;
  key.iy = iy ;
  key.is = is ;

  key.x = x ;
  key.y = y ;
  key.s = s ;
  
  key.sigma = sigma ;
  
  return key ;
}

// ===================================================================
//                                                           process()
// -------------------------------------------------------------------

/** @brief Compute Gaussian Scale Space
 **
 ** The method computes the Gaussian scale space of the specified
 ** image. The scale space data is managed internally and can be
 ** accessed by means of getOctave() and getLevel().
 **
 ** @remark Calling this method will delete the list of keypoints
 ** constructed by detectKeypoints().
 **
 ** @param _im_pt pointer to image data.
 ** @param _width image width.
 ** @param _height image height .
 **/
void
Sift::
process(const pixel_t* _im_pt, int _width, int _height)
{
  using namespace Detail ;

  width  = _width ;
  height = _height ;
  prepareBuffers() ;
  
  VL::float_t sigmak = powf(2.0f, 1.0 / S) ;
  VL::float_t dsigma0 = sigma0 * sqrt (1.0f - 1.0f / (sigmak*sigmak) ) ;
  
  // -----------------------------------------------------------------
  //                                                 Make pyramid base
  // -----------------------------------------------------------------
  if( omin < 0 ) {
    copyAndUpsampleRows(temp,       _im_pt, width,  height  ) ;
    copyAndUpsampleRows(octaves[0], temp,   height, 2*width ) ;      

    for(int o = -1 ; o > omin ; --o) {
      copyAndUpsampleRows(temp,       octaves[0], width  << -o,    height << -o) ;
      copyAndUpsampleRows(octaves[0], temp,       height << -o, 2*(width  << -o)) ;             }

  } else if( omin > 0 ) {
    copyAndDownsample(octaves[0], _im_pt, width, height, 1 << omin) ;
  } else {
    copy(octaves[0], _im_pt, width, height) ;
  }

  {
    VL::float_t sa = sigma0 * powf(sigmak, smin) ; 
    VL::float_t sb = sigman / powf(2.0f,   omin) ; // review this
    if( sa > sb ) {
      VL::float_t sd = sqrt ( sa*sa - sb*sb ) ;
      smooth( octaves[0], temp, octaves[0], 
              getOctaveWidth(omin),
              getOctaveHeight(omin), 
              sd ) ;
    }
  }

  // -----------------------------------------------------------------
  //                                                      Make octaves
  // -----------------------------------------------------------------
  for(int o = omin ; o < omin+O ; ++o) {
    // Prepare octave base
    if( o > omin ) {
      int sbest = std::min(smin + S, smax) ;
      copyAndDownsample(getLevel(o,   smin ), 
				getLevel(o-1, sbest),
				getOctaveWidth(o-1),
				getOctaveHeight(o-1), 2 ) ;
      VL::float_t sa = sigma0 * powf(sigmak, smin      ) ;
      VL::float_t sb = sigma0 * powf(sigmak, sbest - S ) ;
      if(sa > sb ) {
        VL::float_t sd = sqrt ( sa*sa - sb*sb ) ;
        smooth( getLevel(o,0), temp, getLevel(o,0), 
                getOctaveWidth(o), getOctaveHeight(o),
                sd ) ;
      }
    }

    // Make other levels
    for(int s = smin+1 ; s <= smax ; ++s) {
      VL::float_t sd = dsigma0 * powf(sigmak, s) ;
      smooth( getLevel(o,s), temp, getLevel(o,s-1),
              getOctaveWidth(o), getOctaveHeight(o),
              sd ) ;
    }
  }
}

/** @brief Sift detector
 **
 ** The function runs the SIFT detector on the stored Gaussian scale
 ** space (see process()). The detector consists in three steps
 **
 ** - local maxima detection;
 ** - subpixel interpolation;
 ** - rejection of weak keypoints (@a threhsold);
 ** - rejection of keypoints on edge-like structures (@a edgeThreshold).
 **
 ** As they are found, keypoints are added to an internal list.  This
 ** list can be accessed by means of the member functions
 ** getKeypointsBegin() and getKeypointsEnd(). The list is ordered by
 ** octave, which is usefult to speed-up computeKeypointOrientations()
 ** and computeKeypointDescriptor().
 **/



__global__ void DogKernel(pixel_t* dst, pixel_t* srca, pixel_t* srcb, int width)

{

  __shared__ pixel_t src[D_BLOCK_SIZE];

int tx = threadIdx.x;
int bx = blockIdx.x;
int by = blockIdx.y;
int Row = by;
int Col = bx*D_BLOCK_SIZE + tx; //D_BLOCK_SIZE = 128


  src[tx] = srcb[Row * width + Col];

  __syncthreads();

  if (Col < width)
  {
    dst[Row * width + Col] = srca[Row * width + Col] - src[tx];
    srca[Row * width + Col] = src[tx];
  }
  
}



void Sift::Compute_Dog (pixel_t* pt, int o, int smin, int smax, int width, int height)
{

    unsigned int src_size = sizeof(pixel_t) * (width*height);
    unsigned int dst_size = sizeof(pixel_t) * (width*height);   

    pixel_t* dst_d = NULL;
    pixel_t* srca_d = NULL;
    pixel_t* srcb_d = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &dst_d, dst_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &srca_d, src_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &srcb_d, src_size));  

    dim3 dimBlock, dimGrid;
    dimBlock.x = D_BLOCK_SIZE;
    dimBlock.y = 1;   
    dimGrid.x = (width / D_BLOCK_SIZE) + ( (width % D_BLOCK_SIZE) ? 1:0 );
    dimGrid.y = height;  


    pixel_t* srca = getLevel(o, smin) ;
    CUDA_SAFE_CALL( cudaMemcpy(srca_d, srca, src_size, cudaMemcpyHostToDevice) );

    for(int s = smin + 1 ; s <= smax ; ++s) 
    {
     pixel_t* srcb = getLevel(o, s) ;

     CUDA_SAFE_CALL( cudaMemcpy(srcb_d, srcb, src_size, cudaMemcpyHostToDevice) );

     DogKernel<<<dimGrid, dimBlock>>>(dst_d, srca_d, srcb_d, width);

     CUDA_SAFE_CALL(cudaMemcpy(pt, dst_d, dst_size, cudaMemcpyDeviceToHost));

     pt = pt + width*height;
    }


    cudaFree(dst_d);
    cudaFree(srca_d);
    cudaFree(srcb_d);
}

__global__ void FindkKernel(kvalue* dst, pixel_t* srcT, pixel_t* srcM, pixel_t* srcB, int width, int height, float threshold, float edgethreshold)

{

  __shared__ pixel_t Mtop[F_BLOCK_SIZE][F_BLOCK_SIZE];  //F_BLOCK_SIZE = F_TILE_SIZE + 2
  __shared__ pixel_t Mmid[F_BLOCK_SIZE][F_BLOCK_SIZE];
  __shared__ pixel_t Mbot[F_BLOCK_SIZE][F_BLOCK_SIZE];

int tx, ty, bx, by;

 tx = threadIdx.x;
 ty = threadIdx.y;
 bx = blockIdx.x;
 by = blockIdx.y;

int i, j, Row, Col;
float extr = 1.0f;
float Threshold = threshold;
float edgeThreshold = edgethreshold;

	Row = by*F_TILE_SIZE + ty;
	Col = bx*F_TILE_SIZE + tx; 
     if (Row < height && Col < width)
       {
 	Mtop[ty][tx] = srcT[Row * width + Col];
 	Mmid[ty][tx] = srcM[Row * width + Col];
  	Mbot[ty][tx] = srcB[Row * width + Col];
        //dst[Row * width + Col].flag = 0.0f;
       }
    else
       {
 	Mtop[ty][tx] = 0;
 	Mmid[ty][tx] = 0;
  	Mbot[ty][tx] = 0;
       }


     __syncthreads();

 if(ty < F_TILE_SIZE && tx < F_TILE_SIZE && Row < (height -1) && Col < (width-1))  
   { 
      if (Mmid[ty+1][tx+1] > 0)
   
       {
        for(i = 0; i < 3; i++) 
         {
          for(j = 0; j < 3; j++)
             {
             if ( Mmid[ty+1][tx+1] < Mtop[ty+i][tx+j] ||  Mmid[ty+1][tx+1] < Mbot[ty+i][tx+j] ||
		  Mmid[ty+1][tx+1] < Mmid[ty][tx+j]   ||  Mmid[ty+1][tx+1] < Mmid[ty+2][tx+j] ||
 		  Mmid[ty+1][tx+1] < Mmid[ty+1][tx]   ||  Mmid[ty+1][tx+1] < Mmid[ty+1][tx+2] || 
		  Mmid[ty+1][tx+1] < Threshold)
                     {  extr = 0; break; }            
              }
             if (extr == 0)
                  break;
	   }
        }

        else 

        {
        for(i = 0; i < 3; i++) 
         {
          for(j = 0; j < 3; j++)
             {
             if ( Mmid[ty+1][tx+1] > Mtop[ty+i][tx+j] ||  Mmid[ty+1][tx+1] > Mbot[ty+i][tx+j] ||
		  Mmid[ty+1][tx+1] > Mmid[ty][tx+j]   ||  Mmid[ty+1][tx+1] > Mmid[ty+2][tx+j] ||
 		  Mmid[ty+1][tx+1] > Mmid[ty+1][tx]   ||  Mmid[ty+1][tx+1] > Mmid[ty+1][tx+2] || 
		  Mmid[ty+1][tx+1] > Threshold * (-1))
                    {  extr = 0; break; }           
              }
             if (extr == 0)
                  break;
	  }
	 } 

     __syncthreads();
/*
      if(extr == 1)
          { 
	    //float4 value = RefineKernel(Mtop, Mmid, Mbot, width, height, threshold, edgethreshold)	


    	//int StepX = 0;
   	//int StepY = 0;
    	float ds = 0.0f;
    	float dy = 0.0f;
    	float dx = 0.0f;
	float Vx2, fx, fy, fs, fxx, fyy, fss, fxy, fxs, fys;
   // for(int iter = 0 ; iter < 5 ; ++iter) {
    
	//tx = threadIdx.x + StepX;
	//ty = threadIdx.y + StepY;
	
	 Vx2 = Mmid[ty+1][tx+1] * 2.0f;
	 fx = 0.5f * (Mmid[ty+1][tx+2] - Mmid[ty+1][tx]);
	 fy = 0.5f * (Mmid[ty+2][tx+1] - Mmid[ty][tx+1]);
	 fs = 0.5f * (Mbot[ty+1][tx+1] - Mtop[ty+1][tx+1]);

	 fxx = Mmid[ty+1][tx+2] + Mmid[ty+1][tx] - Vx2;
	 fyy = Mmid[ty+2][tx+1] + Mmid[ty][tx+1] - Vx2;
	 fss = Mbot[ty+1][tx+1] + Mtop[ty+1][tx+1] - Vx2;

	 fxy = 0.25f * (Mmid[ty+2][tx+2] + Mmid[ty][tx] - Mmid[ty+2][tx] - Mmid[ty][tx+2]);
	 fxs = 0.25f * (Mbot[ty+1][tx+2] + Mtop[ty+1][tx] - Mbot[ty+1][tx] - Mtop[ty+1][tx+2]);
	 fys = 0.25f * (Mbot[ty+2][tx+1] + Mtop[ty][tx+1] - Mbot[ty][tx+1] - Mtop[ty+2][tx+1]);

				//need to solve dx, dy, ds;
				// |-fx|     | fxx fxy fxs |   |dx|
				// |-fy|  =  | fxy fyy fys | * |dy|
				// |-fs|     | fxs fys fss |   |ds|

	float4 A0 = fxx > 0? make_float4(fxx, fxy, fxs, -fx) : make_float4(-fxx, -fxy, -fxs, fx);
	float4 A1 = fxy > 0? make_float4(fxy, fyy, fys, -fy) : make_float4(-fxy, -fyy, -fys, fy);
	float4 A2 = fxs > 0? make_float4(fxs, fys, fss, -fs) : make_float4(-fxs, -fys, -fss, fs);
	float maxa = max(max(A0.x, A1.x), A2.x);

	 if(maxa >= 1e-10){
	     if(maxa == A1.x){
		    float4 TEMP = A1; A1 = A0; A0 = TEMP;
		 }else if(maxa == A2.x){
		     float4 TEMP = A2; A2 = A0; A0 = TEMP;
		 }
		 A0.y /= A0.x;	A0.z /= A0.x;	A0.w/= A0.x;
		 A1.y -= A1.x * A0.y;	A1.z -= A1.x * A0.z;	A1.w -= A1.x * A0.w;
		 A2.y -= A2.x * A0.y;	A2.z -= A2.x * A0.z;	A2.w -= A2.x * A0.w;

	     if(abs(A2.y) > abs(A1.y)){
		      float4 TEMP = A2;	A2 = A1; A1 = TEMP;
		  }

	     if(abs(A1.y) >= 1e-10) {
		  A1.z /= A1.y;	A1.w /= A1.y;
		  A2.z -= A2.y * A1.z;	A2.w -= A2.y * A1.w;
		    if(abs(A2.z) >= 1e-10) {
			 ds = A2.w / A2.z;
			 dy = A1.w - ds * A1.z;
			 dx = A0.w - ds * A0.z - dy * A0.y;
		    }
	       }
	   }


          // StepX= ((ds >  0.6 && ( bx*F_TILE_SIZE + tx + 1  ) < width -2) ?  1 : 0 ) + ((ds < -0.6 && (bx*F_TILE_SIZE + tx + 1) > 1   ) ? -1 : 0 ) ;
         
          // StepY= ((dy >  0.6 && ( by*F_TILE_SIZE + ty + 1 )< height -2) ?  1 : 0 ) + ((dy < -0.6 && (by*F_TILE_SIZE + ty + 1) > 1   ) ? -1 : 0 ) ;

        //  if( StepX == 0 && StepY == 0 ) break ;

   // }
	
	float val = Mmid[ty+1][tx+1] + 0.5f * (fx * dx + fy * dy + fs * ds);
	float score = (fxx + fyy) * (fxx + fyy) / (fxx * fyy - fxy * fxy);
	

        if(fabs(val) > threshold && score < (edgeThreshold + 1)*(edgeThreshold + 1)/edgeThreshold && score >= 0 &&
            fabs(dx) < 1.5 && fabs(dy) < 0.6 && fabs(ds) < 0.6 )
					  	{       
	    					dst[(Row+1) * width + (Col+1)].dx = dx;  
	   					dst[(Row+1) * width + (Col+1)].dy = dy;
	    					dst[(Row+1) * width + (Col+1)].ds = ds; 
	    					dst[(Row+1) * width + (Col+1)].flag = extr;  
          					} 
	else
	    dst[(Row+1) * width + (Col+1)].flag = 0.0f;	*/	  
	//}       
	//else
	    dst[(Row+1) * width + (Col+1)].flag = extr;   
    }

   /*  if (Row < height && Col < width){
	srcT[Row * width + Col] = Mmid[ty][tx];
	srcM[Row * width + Col] = Mbot[ty][tx];
	}
   */

}
  
void 
Sift :: FindkExtrem(pixel_t* pt, int width, int height, int o, float xperiod, float threshold, float edgethreshold)
{
	
    unsigned int dst_size = sizeof(kvalue) * (width*height);
    unsigned int src_size = sizeof(pixel_t) * (width*height);


    pixel_t* srcT_d = NULL;
    pixel_t* srcM_d = NULL;
    pixel_t* srcB_d = NULL;
    kvalue* dst_d = NULL;
    
    CUDA_SAFE_CALL( cudaMalloc( (void**) &srcT_d, src_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &srcM_d, src_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &srcB_d, src_size));  
    CUDA_SAFE_CALL( cudaMalloc( (void**) &dst_d, dst_size));  
 

    dim3 dimBlock, dimGrid;
    dimBlock.x = F_BLOCK_SIZE;
    dimBlock.y = F_BLOCK_SIZE;   
    dimGrid.x = ((width-2) / F_TILE_SIZE) + (((width-2) % F_TILE_SIZE) ? 1:0 );
    dimGrid.y = ((height-2) / F_TILE_SIZE) + (((height-2) % F_TILE_SIZE) ? 1:0 ); 


    pixel_t* src = pt;
   // kvalue* dst;
   // dst = new kvalue [width*height] ;
    Keypoint k ;

   // CUDA_SAFE_CALL( cudaMemcpy(srcT_d, src, src_size, cudaMemcpyHostToDevice) );
  //  src = src + width * height;
    //CUDA_SAFE_CALL( cudaMemcpy(srcM_d, src, src_size, cudaMemcpyHostToDevice) );
      int uu = 0;
    for(int s = smin + 1 ; s <= smax-2 ; ++s) 
    {
      CUDA_SAFE_CALL( cudaMemcpy(srcT_d, src, src_size, cudaMemcpyHostToDevice) );
      src = src + width * height;
      CUDA_SAFE_CALL( cudaMemcpy(srcM_d, src, src_size, cudaMemcpyHostToDevice) );
      src = src + width * height;
      CUDA_SAFE_CALL( cudaMemcpy(srcB_d, src, src_size, cudaMemcpyHostToDevice) );
      src = src - width * height;

	
      FindkKernel<<<dimGrid, dimBlock>>>(dst_d, srcT_d, srcM_d, srcB_d, width, height, 0.8*threshold, edgethreshold);

      CUDA_SAFE_CALL(cudaMemcpy(Kmid, dst_d, dst_size, cudaMemcpyDeviceToHost));


     // float xn;
     // float yn;
     // float sn;
     for(int y = 0; y < height-1; y++)
       for(int x = 0; x < width-1; x++)
	{
	  if (Kmid[width * y + x].flag == 1.0f)
	   {   
	      
	     /*
              xn = x + Kmid[width * y + x].dx;
              yn = y + Kmid[width * y + x].dy;
              sn = s + Kmid[width * y + x].ds;
             if(xn >= 0 && xn <= width -1 && yn >= 0 && yn <= height -1 && sn >= smin && sn <= smax ) 
               {  */
		 k.o = o;
                 k.ix = x ;
             	 k.iy = y ;
            	 k.is = s ;
              	// k.x = xn * xperiod ; 
            	// k.y = yn * xperiod ; 
           	// k.s = sn; 

            	// k.sigma = getScaleFromIndex(o,sn) ;
                 keypoints.push_back(k);
		// KeyNum[o-omin]++;
	         uu++; 
                // std::cout<<x<<","<<y<<","<<s<<","<<k.x<<","<<k.y<<","<<k.sigma<<","<<"||   "<<std::flush; 
	      // }
	     }

	  }
    }
     	      std::cout<<" " <<" "<<std::endl ; 
	      //std::cout<<"o is "<<o<<"   total key number is "<<KeyNum[o-omin]<<std::endl;


    cudaFree(srcT_d);
    cudaFree(srcM_d);
    cudaFree(srcB_d);
    cudaFree(dst_d);
    //free dst;


}

void
Sift::detectKeypoints(VL::float_t threshold, VL::float_t edgeThreshold)
{
  keypoints.clear() ;

  int nValidatedKeypoints = 0 ;

  // Process one octave per time
  for(int o = omin; o < omin + O; ++o) {
        
    int const xo = 1 ;
    int const yo = getOctaveWidth(o) ;
    int const so = getOctaveWidth(o) * getOctaveHeight(o) ;
    int const ow = getOctaveWidth(o) ;
    int const oh = getOctaveHeight(o) ;

    VL::float_t xperiod = getOctaveSamplingPeriod(o) ;

    // -----------------------------------------------------------------
    //                                           Difference of Gaussians
    // -----------------------------------------------------------------
    pixel_t* dog = temp ;
    tempIsGrad = false ;
    KeyNum[o-omin] = 0;
    {
      pixel_t* pt = dog ;

     // Compute_Dog (pt, o, smin, smax, yo, oh);   //gpu function

       
      for(int s = smin ; s <= smax-1 ; ++s) {
        pixel_t* srca = getLevel(o, s  ) ;
        pixel_t* srcb = getLevel(o, s+1) ;
        pixel_t* enda = srcb ;
        while( srca != enda ) {
          *pt++ = *srcb++ - *srca++ ;
        }
      }
       
    }
    
    // -----------------------------------------------------------------
    //                                           Find points of extremum
    // -----------------------------------------------------------------
    // std::cout<<" " <<" "<<std::endl ; 
     std::cout<<" " <<" "<<std::endl ; 
     //std::cout<<"O is  "<<o<<" "<<std::endl ; 

    // pixel_t* pt = dog ;
    // if (O < 8)
    // FindkExtrem_small(pt, yo, oh, o, xperiod, threshold, edgeThreshold);
    // else 
    // FindkExtrem(pt, yo, oh, o, xperiod, threshold, edgeThreshold);



    {
      int uu;
      pixel_t* pt  = dog + xo + yo + so ;
      for(int s = smin+1 ; s <= smax-2 ; ++s) {
        for(int y = 1 ; y < oh - 1 ; ++y) {
          for(int x = 1 ; x < ow - 1 ; ++x) {          
            pixel_t v = *pt ;
            
            // assert( (pt - x*xo - y*yo - (s-smin)*so) - dog == 0 ) ;
            
#define CHECK_NEIGHBORS(CMP,SGN)                    \
            ( v CMP ## = SGN 0.8 * threshold &&     \
              v CMP *(pt + xo) &&                   \
              v CMP *(pt - xo) &&                   \
              v CMP *(pt + so) &&                   \
              v CMP *(pt - so) &&                   \
              v CMP *(pt + yo) &&                   \
              v CMP *(pt - yo) &&                   \
                                                    \
              v CMP *(pt + yo + xo) &&              \
              v CMP *(pt + yo - xo) &&              \
              v CMP *(pt - yo + xo) &&              \
              v CMP *(pt - yo - xo) &&              \
                                                    \
              v CMP *(pt + xo      + so) &&         \
              v CMP *(pt - xo      + so) &&         \
              v CMP *(pt + yo      + so) &&         \
              v CMP *(pt - yo      + so) &&         \
              v CMP *(pt + yo + xo + so) &&         \
              v CMP *(pt + yo - xo + so) &&         \
              v CMP *(pt - yo + xo + so) &&         \
              v CMP *(pt - yo - xo + so) &&         \
                                                    \
              v CMP *(pt + xo      - so) &&         \
              v CMP *(pt - xo      - so) &&         \
              v CMP *(pt + yo      - so) &&         \
              v CMP *(pt - yo      - so) &&         \
              v CMP *(pt + yo + xo - so) &&         \
              v CMP *(pt + yo - xo - so) &&         \
              v CMP *(pt - yo + xo - so) &&         \
              v CMP *(pt - yo - xo - so) )
            
            if( CHECK_NEIGHBORS(>,+) || CHECK_NEIGHBORS(<,-) ) {
              
              Keypoint k ;
              k.ix = x ;
              k.iy = y ;
              k.is = s ;
              keypoints.push_back(k) ;
		uu++;
            }
            pt += 1 ;
          }
          pt += 2 ;
        }
        pt += 2*yo ;

       uu = 0; 
      }
    }
  


    // -----------------------------------------------------------------
    //                                               Refine local maxima
    // -----------------------------------------------------------------
  // int uu;
    { // refine
      KeypointsIter siter ;
      KeypointsIter diter ;

      
      for(diter = siter = keypointsBegin() + nValidatedKeypoints ; 
          siter != keypointsEnd() ; 
          ++siter) {
       
        int x = int( siter->ix ) ;
        int y = int( siter->iy ) ;
        int s = int( siter->is ) ;
                
        VL::float_t Dx=0,Dy=0,Ds=0,Dxx=0,Dyy=0,Dss=0,Dxy=0,Dxs=0,Dys=0 ;
        VL::float_t  b [3] ;
        pixel_t* pt ;
        int dx = 0 ;
        int dy = 0 ;

        // must be exec. at least once
        for(int iter = 0 ; iter < 5 ; ++iter) {

          VL::float_t A[3*3] ;          

          x += dx ;
          y += dy ;

          pt = dog 
            + xo * x
            + yo * y
            + so * (s - smin) ;

#define at(dx,dy,ds) (*( pt + (dx)*xo + (dy)*yo + (ds)*so))
#define Aat(i,j)     (A[(i)+(j)*3])    
          
  
          Dx = 0.5 * (at(+1,0,0) - at(-1,0,0)) ;
          Dy = 0.5 * (at(0,+1,0) - at(0,-1,0));
          Ds = 0.5 * (at(0,0,+1) - at(0,0,-1)) ;
          
          // Compute the Hessian. 
          Dxx = (at(+1,0,0) + at(-1,0,0) - 2.0 * at(0,0,0)) ;
          Dyy = (at(0,+1,0) + at(0,-1,0) - 2.0 * at(0,0,0)) ;
          Dss = (at(0,0,+1) + at(0,0,-1) - 2.0 * at(0,0,0)) ;
          
          Dxy = 0.25 * ( at(+1,+1,0) + at(-1,-1,0) - at(-1,+1,0) - at(+1,-1,0) ) ;
          Dxs = 0.25 * ( at(+1,0,+1) + at(-1,0,-1) - at(-1,0,+1) - at(+1,0,-1) ) ;
          Dys = 0.25 * ( at(0,+1,+1) + at(0,-1,-1) - at(0,-1,+1) - at(0,+1,-1) ) ;
          
          // Solve linear system. 
          Aat(0,0) = Dxx ;
          Aat(1,1) = Dyy ;
          Aat(2,2) = Dss ;
          Aat(0,1) = Aat(1,0) = Dxy ;
          Aat(0,2) = Aat(2,0) = Dxs ;
          Aat(1,2) = Aat(2,1) = Dys ;
          
          b[0] = - Dx ;
          b[1] = - Dy ;
          b[2] = - Ds ;
          
          // Gauss elimination
          for(int j = 0 ; j < 3 ; ++j) {

            // look for leading pivot
            VL::float_t maxa = 0 ;
            VL::float_t maxabsa = 0 ;
            int   maxi = -1 ;
            int i ;
            for(i = j ; i < 3 ; ++i) {
              VL::float_t a    = Aat(i,j) ;
              VL::float_t absa = fabsf( a ) ;
              if ( absa > maxabsa ) {
                maxa    = a ;
                maxabsa = absa ;
                maxi    = i ;
              }
            }

            // singular?
            if( maxabsa < 1e-10f ) {
              b[0] = 0 ;
              b[1] = 0 ;
              b[2] = 0 ;
              break ;
            }

            i = maxi ;

            // swap j-th row with i-th row and
            // normalize j-th row
            for(int jj = j ; jj < 3 ; ++jj) {
              std::swap( Aat(j,jj) , Aat(i,jj) ) ;
              Aat(j,jj) /= maxa ;
            }
            std::swap( b[j], b[i] ) ;
            b[j] /= maxa ;

            // elimination
            for(int ii = j+1 ; ii < 3 ; ++ii) {
              VL::float_t x = Aat(ii,j) ;
              for(int jj = j ; jj < 3 ; ++jj) {
                Aat(ii,jj) -= x * Aat(j,jj) ;                
              }
              b[ii] -= x * b[j] ;
            }
          }

          // backward substitution
          for(int i = 2 ; i > 0 ; --i) {
            VL::float_t x = b[i] ;
            for(int ii = i-1 ; ii >= 0 ; --ii) {
              b[ii] -= x * Aat(ii,i) ;
            }
          }

          // If the translation of the keypoint is big, move the keypoint
           // and re-iterate the computation. Otherwise we are all set.
           
          dx= ((b[0] >  0.6 && x < ow-2) ?  1 : 0 )
            + ((b[0] < -0.6 && x > 1   ) ? -1 : 0 ) ;
          
          dy= ((b[1] >  0.6 && y < oh-2) ?  1 : 0 )
            + ((b[1] < -0.6 && y > 1   ) ? -1 : 0 ) ;

              
        //  std::cout<<x<<","<<y<<"="<<at(0,0,0) <<"(" <<at(0,0,0)+0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2])<<")" <<" "<<std::flush ; 
          

          if( dx == 0 && dy == 0 ) break ;
        }
        

        
        // Accept-reject keypoint
        {
          VL::float_t val = at(0,0,0) + 0.5 * (Dx * b[0] + Dy * b[1] + Ds * b[2]) ; 
          VL::float_t score = (Dxx+Dyy)*(Dxx+Dyy) / (Dxx*Dyy - Dxy*Dxy) ; 
          VL::float_t xn = x + b[0] ;
          VL::float_t yn = y + b[1] ;
          VL::float_t sn = s + b[2] ;
          
          if(fast_abs(val) > threshold &&
             score < (edgeThreshold+1)*(edgeThreshold+1)/edgeThreshold && 
             score >= 0 &&
             fast_abs(b[0]) < 1.5 &&
             fast_abs(b[1]) < 1.5 &&
             fast_abs(b[2]) < 1.5 &&
             xn >= 0    &&
             xn <= ow-1 &&
             yn >= 0    &&
             yn <= oh-1 &&
             sn >= smin &&
             sn <= smax ) 
     {
            
            diter->o  = o ;
       
            diter->ix = x ;
            diter->iy = y ;
            diter->is = s ;

            diter->x = xn * xperiod ; 
            diter->y = yn * xperiod ; 
            diter->s = sn ;

            diter->sigma = getScaleFromIndex(o,sn) ;

            ++diter ;
             
	  // std::cout<<x<<","<<y<<","<<s<<","<<o<<","<<" "<<std::flush; 
            		 KeyNum[o-omin]++;


          }
        }
      } // next candidate keypoint

      // prepare for next octave
      keypoints.resize( diter - keypoints.begin() ) ;
      nValidatedKeypoints = keypoints.size() ;
    } // refine block

	      std::cout<<"o is "<<o<<"   total key number is "<<KeyNum[o-omin]<<std::endl;
       
    } // next octave 
}



// ===================================================================
//                                       computeKeypointOrientations()
// -------------------------------------------------------------------

/** @brief Compute modulus and phase of the gradient
 **
 ** The function computes the modulus and the angle of the gradient of
 ** the specified octave @a o. The result is stored in a temporary
 ** internal buffer accessed by computeKeypointDescriptor() and
 ** computeKeypointOrientations().
 **
 ** The SIFT detector provides keypoint with scale index s in the
 ** range @c smin+1 and @c smax-2. As such, the buffer contains only
 ** these levels.
 **
 ** If called mutliple time on the same data, the function exits
 ** immediately.
 **
 ** @param o octave of interest.
 **/


__global__ void GradKernelZ(pixel_t* src, pixel_t* dst, int width, int height, int square)

{
 __shared__ pixel_t Ms[G_BLOCK_SIZE][G_BLOCK_SIZE]; //

int tx, ty, bx, by, bz;
//float m, t;

   tx = threadIdx.x;
   ty = threadIdx.y;
   bx = blockIdx.x;
   by = blockIdx.y;
   bz = blockIdx.z;

int Row = by*G_TILE_SIZE + ty;
int Col = bx*G_TILE_SIZE + tx; 
int Dep = bz*square;

     if (Row < height && Col < width)
       {
 	Ms[ty][tx] = src[Dep + Row * width + Col];
       }
    else
       {
 	Ms[ty][tx] = 0.0f;
       }

     __syncthreads();

 if(ty < G_TILE_SIZE && tx < G_TILE_SIZE && Row < (height -1) && Col < (width-1))  
	{
	float_t Gx = 0.5f * (Ms[ty+1][tx+2] - Ms[ty+1][tx]);
	float_t Gy = 0.5f * (Ms[ty+2][tx+1] - Ms[ty][tx+1]);
  	float_t m = sqrt( Gx*Gx + Gy*Gy );
 	float_t x = atan2(Gy, Gx) + float(2*M_PI);
	float_t t = (x >= 0)? fmod (x, float(2*M_PI)) : float(2*M_PI) + fmod (x, float(2*M_PI));

	dst[2*Dep + 2*width*(Row + 1) + 2*(Col + 1)] = m;
	dst[2*Dep + 2*width*(Row + 1) + 2*(Col + 1) + 1] = t;
	}

}



__global__ void GradKernel(pixel_t* src, pixel_t* dst, int width, int height, int square)

{
 __shared__ pixel_t Ms[G_BLOCK_SIZE][G_BLOCK_SIZE]; //F_BLOCK_SIZE = F_TILE_SIZE + 2

int tx, ty, bx, by;
//float m, t;

   tx = threadIdx.x;
   ty = threadIdx.y;
   bx = blockIdx.x;
   by = blockIdx.y;


int Row = by*G_TILE_SIZE + ty;
int Col = bx*G_TILE_SIZE + tx; 


     if (Row < height && Col < width)
       {
 	Ms[ty][tx] = src[ Row * width + Col];
       }
    else
       {
 	Ms[ty][tx] = 0.0f;
       }

     __syncthreads();

 if(ty < G_TILE_SIZE && tx < G_TILE_SIZE && Row < (height -1) && Col < (width-1))  
	{
	float_t Gx = 0.5f * (Ms[ty+1][tx+2] - Ms[ty+1][tx]);
	float_t Gy = 0.5f * (Ms[ty+2][tx+1] - Ms[ty][tx+1]);
  	float_t m = sqrt( Gx*Gx + Gy*Gy );
 	float_t x = atan2(Gy, Gx) + float(2*M_PI);
	float_t t = (x >= 0)? fmod (x, float(2*M_PI)) : float(2*M_PI) + fmod (x, float(2*M_PI));

	dst[ 2*width*(Row + 1) + 2*(Col + 1)] = m;
	dst[ 2*width*(Row + 1) + 2*(Col + 1) + 1] = t;
	}

}
void 
Sift::GradinGpu(pixel_t* pt, int o, int width, int height)
{

    //int S = smax - smin - 2;
    int square = width * height;
    //unsigned int dst_size = sizeof(pixel_t) * (2*S*width*height);
   // unsigned int src_size = sizeof(pixel_t) * (S*width*height);	
    unsigned int dst_size = sizeof(pixel_t) * (2*width*height);
   unsigned int src_size = sizeof(pixel_t) * (width*height);	

    pixel_t* src_d = NULL;
    pixel_t* dst_d = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &src_d, src_size));  
    CUDA_SAFE_CALL( cudaMalloc( (void**) &dst_d, dst_size));

    dim3 dimBlock, dimGrid;
    dimBlock.x = G_BLOCK_SIZE;
    dimBlock.y = G_BLOCK_SIZE;   
    dimGrid.x = ((width-2) / G_TILE_SIZE) + (((width-2) % G_TILE_SIZE) ? 1:0 );
    dimGrid.y = ((height-2) / G_TILE_SIZE) + (((height-2) % G_TILE_SIZE) ? 1:0 ); 
  //dimGrid.z = S;
    dimGrid.z = 1;
 

   for(int s = smin + 1 ; s <= smax-2 ; ++s) 
    {
      pixel_t* src  = getLevel(o, s);

      CUDA_SAFE_CALL( cudaMemcpy(src_d, src, src_size, cudaMemcpyHostToDevice) );
	
      GradKernel<<<dimGrid, dimBlock>>>(src_d, dst_d, width, height, square);

      CUDA_SAFE_CALL(cudaMemcpy(pt, dst_d, dst_size, cudaMemcpyDeviceToHost));

      pt = pt + 2*width*height;
	
     }

    cudaFree(src_d);
    cudaFree(dst_d);

}


void
Sift::prepareGrad(int o)
{ 
  int const ow = getOctaveWidth(o) ;
  int const oh = getOctaveHeight(o) ;
  //int const xo = 1 ;
  int const yo = ow ;
  //int const so = oh*ow ;

  if( ! tempIsGrad || tempOctave != o ) {
/*
    // compute dx/dy
    for(int s = smin+1 ; s <= smax-2 ; ++s) {
      for(int y = 1 ; y < oh-1 ; ++y ) {
        pixel_t* src  = getLevel(o, s) + xo + yo*y ;        
        pixel_t* end  = src + ow - 1 ;
        pixel_t* grad = 2 * (xo + yo*y + (s - smin -1)*so) + temp ;
        while(src != end) {
          VL::float_t Gx = 0.5 * ( *(src+xo) - *(src-xo) ) ;
          VL::float_t Gy = 0.5 * ( *(src+yo) - *(src-yo) ) ;
          VL::float_t m = fast_sqrt( Gx*Gx + Gy*Gy ) ;
          VL::float_t t = fast_mod_2pi( fast_atan2(Gy, Gx) + VL::float_t(2*M_PI) );
          *grad++ = pixel_t( m ) ;
          *grad++ = pixel_t( t ) ;
          ++src ;
        }
      }
    }  
*/
        pixel_t* grad = temp;	
	GradinGpu(grad, o, yo, oh);
  }

  tempIsGrad = true ;
  tempOctave = o ;
}


__device__ void  normalize_histogram(float* L_begin, float* L_end)
{
  float* L_iter ;
  float norm = 0.0f ;

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
    norm += (*L_iter) * (*L_iter) ;

  norm = sqrt(norm) ;

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
	*L_iter /= norm;
   // *L_iter /= (norm + numeric_limits<float>::epsilon() ) ;
}

__global__ void GetkKernel(Sift::Keypoint* Kin, pixel_t* Grad, OKvalue* Kout, int Klength, int width, int height, int smin, float xperiod, int magnif)
{
	int i;
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int index = bx * K_BLOCK_SIZE + tx;
	VL::float_t angles [4];

	int nbins = 36;
	//VL::float_t WinFactor = 1.5f;
	VL::float_t hist[36];

	int ow = width;
	int oh = height;
	int xo = 2;
	int yo = xo * ow;
	int so = yo * oh;

   if (index < Klength){

	VL::float_t x = Kin[index].x / xperiod;
	VL::float_t y = Kin[index].y / xperiod;
	VL::float_t sigma = Kin[index].sigma / xperiod;

  	int xi = ((int) (x+0.5)) ; 
  	int yi = ((int) (y+0.5)) ;
  	int si = Kin[index].is ;
	VL::float_t sigmaw = 1.50f * sigma;  //winFactor
  	int Wo = (int) floor(3.0 * sigmaw);

	int NBO = 8;
	int NBP = 4;
	VL::float_t SBP = magnif*sigma;
	int Wd = (int) floor (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;
	int binto = 1;
	int binyo = NBO*NBP;
	int binxo = NBO;
	int bin;
	
	

	for (i = 0; i < nbins; i++)
		hist[i] = 0.0f;

  	pixel_t* pt = Grad + xi * xo + yi * yo + (si - smin -1) * so ;

  	for(int ys = max(-Wo, 1-yi) ; ys <= min(+Wo, oh -2 -yi) ; ++ys) {
   	   for(int xs = max(-Wo, 1-xi) ; xs <= min(+Wo, ow -2 -xi) ; ++xs) {

		VL::float_t dx = xi + xs - x;
      		VL::float_t dy = yi + ys - y;
      		VL::float_t r2 = dx*dx + dy*dy ;
 
      		if(r2 >= Wo*Wo+0.5) continue ;

      		VL::float_t wgt = exp(-(r2 / (2*sigmaw*sigmaw))) ;
      		VL::float_t mod = *(pt + xs*xo + ys*yo) ;
      		VL::float_t ang = *(pt + xs*xo + ys*yo + 1) ;

      		int bin = (int) floor( nbins * ang / (2*M_PI) ) ;
      		hist[bin] += mod * wgt ;        
    	      }
  	   }

#if defined VL_LOWE_STRICT
  // Lowe's version apparently has a little issue with orientations
  // around + or - pi, which we reproduce here for compatibility
  	for (int iter = 0; iter < 6; iter++) {
    		VL::float_t prev  = hist[nbins/2] ;
    	    for (int i = nbins/2-1; i >= -nbins/2 ; --i) {
      			int j  = (i     + nbins) % nbins ;
      			int jp = (i - 1 + nbins) % nbins ;
      			VL::float_t newh = (prev + hist[j] + hist[jp]) / 3.0;
      			prev = hist[j] ;
      			hist[j] = newh ;
    		}
  	     }
#else
  // this is slightly more correct
  	for (int iter = 0; iter < 6; iter++) {
    		VL::float_t prev  = hist[nbins-1] ;
    		VL::float_t first = hist[0] ;   
    		for (i = 0; i < nbins - 1; i++) {
      		VL::float_t newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
      		prev = hist[i] ;
      		hist[i] = newh ;
    		}
    		hist[i] = (prev + hist[i] + first)/3.0 ;
  	}
#endif


		//VL::float_t maxh = * std::max_element(hist, hist + nbins) ;
		VL::float_t maxh = 0;
		for (int i = 0; i < nbins; i++)
		    maxh = max(maxh, hist[i]);


  		int nangles = 0 ;
  		for(int i = 0 ; i < nbins ; ++i) {
    			VL::float_t h0 = hist [i] ;
    			VL::float_t hm = hist [(i-1+nbins) % nbins] ;
    			VL::float_t hp = hist [(i+1+nbins) % nbins] ;    
    			// is this a peak?
    		    if( h0 > 0.8*maxh && h0 > hm && h0 > hp ){
      
      				VL::float_t di = -0.5 * (hp - hm) / (hp+hm-2*h0) ; 
      				VL::float_t th = 2*M_PI * (i+di+0.5) / nbins ;      
      				angles [ nangles ] = th ;
				Kout[index].th[nangles] = th;
				nangles++;
      				if( nangles == 4 )
      				    break;
    			}
  		   }
		 Kout[index].nangles = nangles;

////**************descriptor section******************//

     for(int a = 0 ; a < nangles ; ++a) {

		VL::float_t descr_pt[128];

		  for (int i = 0; i < 128; i ++)
			descr_pt[i] = 0.0f;

		VL::float_t* dpt = descr_pt + (NBP/2) * binyo + (NBP/2) * binxo;

  		VL::float_t st0   = sinf( angles[a] ) ;
 		VL::float_t ct0   = cosf( angles[a] ) ;

#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)
      

  	for(int dyi = max(-Wd, 1-yi) ; dyi <= min(+Wd, oh-2-yi) ; ++dyi) {
    		for(int dxi = max(-Wd, 1-xi) ; dxi <= min(+Wd, ow-2-xi) ; ++dxi) {
      

      		VL::float_t mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      		VL::float_t angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
		//VL::float_t x =  (angles[a] - angle) ;
      		VL::float_t theta = ((angles[a] - angle) >= 0)? fmod ((angles[a] - angle), float(2*M_PI)) : float(2*M_PI) + fmod ((angles[a] - angle), float(2*M_PI)); // lowe compatible ?
      
     
      		VL::float_t dx = xi + dxi - x;
      		VL::float_t dy = yi + dyi - y;
      
      // get the displacement normalized w.r.t. the keypoint
      // orientation and extension.
      		VL::float_t nx = ( ct0 * dx + st0 * dy) / SBP ;
      		VL::float_t ny = (-st0 * dx + ct0 * dy) / SBP ; 
      		VL::float_t nt = NBO * theta / (2*M_PI) ;
      
      // Get the gaussian weight of the sample. The gaussian window
      // has a standard deviation equal to NBP/2. Note that dx and dy
      // are in the normalized frame, so that -NBP/2 <= dx <= NBP/2.
      		VL::float_t const wsigma = NBP/2 ;
      		VL::float_t win = exp(-((nx*nx + ny*ny)/(2.0 * wsigma * wsigma))) ;
      
      // The sample will be distributed in 8 adjacent bins.
      // We start from the ``lower-left'' bin.
      		int binx = floor( nx - 0.5 ) ;
     		int biny = floor( ny - 0.5 ) ;
      		int bint = floor( nt ) ;
      		VL::float_t rbinx = nx - (binx+0.5) ;
      		VL::float_t rbiny = ny - (biny+0.5) ;
      		VL::float_t rbint = nt - bint ;
      		int dbinx ;
      		int dbiny ;
      		int dbint ;

      // Distribute the current sample into the 8 adjacent bins
      			for(dbinx = 0 ; dbinx < 2 ; ++dbinx) 
        			for(dbiny = 0 ; dbiny < 2 ; ++dbiny)
          				for(dbint = 0 ; dbint < 2 ; ++dbint)             
            					if( binx+dbinx >= -(NBP/2) && binx+dbinx <   (NBP/2) && biny+dbiny >= -(NBP/2) && biny+dbiny <   (NBP/2) ) {
              	 VL::float_t weight = win * mod * abs (1 - dbinx - rbinx) * abs (1 - dbiny - rbiny) * abs (1 - dbint - rbint) ;
              	  atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            					}
          			 	           
        			
      				
    		}  
  	    }


  		//if( normalizeDescriptor ) {
     
    			normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;
    			for(bin = 0; bin < NBO*NBP*NBP ; ++bin) {
      			if (descr_pt[bin] > 0.2) descr_pt[bin] = 0.2;
    		        }
    		normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;
  		//}

	   for (int i = 0; i < 128; i ++)
		Kout[index].descr_pt[a*128 + i ] = descr_pt[i]; 

          }
     }
}





/** @brief Compute the orientation(s) of a keypoint
 **
 ** The function computes the orientation of the specified keypoint.
 ** The function returns up to four different orientations, obtained
 ** as strong peaks of the histogram of gradient orientations (a
 ** keypoint can theoretically generate more than four orientations,
 ** but this is very unlikely).
 **
 ** @remark The function needs to compute the gradient modululs and
 ** orientation of the Gaussian scale space octave to which the
 ** keypoint belongs. The result is cached, but discarded if different
 ** octaves are visited. Thereofre it is much quicker to evaluate the
 ** keypoints in their natural octave order.
 **
 ** The keypoint must lie within the scale space. In particular, the
 ** scale index is supposed to be in the range @c smin+1 and @c smax-1
 ** (this is from the SIFT detector). If this is not the case, the
 ** computation is silently aborted and no orientations are returned.
 **
 ** @param angles buffers to store the resulting angles.
 ** @param keypoint keypoint to process.
 ** @return number of orientations found.
 **/
int
Sift::computeKeypointOrientations(VL::float_t angles [4], Keypoint keypoint)
{
  int const   nbins = 36 ;
  VL::float_t const winFactor = 1.5 ;
  VL::float_t hist [nbins] ;

  // octave
  int o = keypoint.o ;
  VL::float_t xperiod = getOctaveSamplingPeriod(o) ;

  // offsets to move in the Gaussian scale space octave
  const int ow = getOctaveWidth(o) ;
  const int oh = getOctaveHeight(o) ;
  const int xo = 2 ;
  const int yo = xo * ow ;
  const int so = yo * oh ;

  // keypoint fractional geometry
  VL::float_t x     = keypoint.x / xperiod ;
  VL::float_t y     = keypoint.y / xperiod ;
  VL::float_t sigma = keypoint.sigma / xperiod ;
  
  // shall we use keypoints.ix,iy,is here?
  int xi = ((int) (x+0.5)) ; 
  int yi = ((int) (y+0.5)) ;
  int si = keypoint.is ;
  
  VL::float_t const sigmaw = winFactor * sigma ;
  int W = (int) floor(3.0 * sigmaw) ;
  
  // skip the keypoint if it is out of bounds
  if(o  < omin   ||
     o  >=omin+O ||
     xi < 0      || 
     xi > ow-1   || 
     yi < 0      || 
     yi > oh-1   || 
     si < smin+1 || 
     si > smax-2 ) {
    std::cerr<<"!"<<std::endl ;
    return 0 ;
  }
  
  // make sure that the gradient buffer is filled with octave o
  prepareGrad(o) ;

  // clear the SIFT histogram
  std::fill(hist, hist + nbins, 0) ;

  // fill the SIFT histogram
  pixel_t* pt = temp + xi * xo + yi * yo + (si - smin -1) * so ;

#undef at
#define at(dx,dy) (*(pt + (dx)*xo + (dy)*yo))

  for(int ys = std::max(-W, 1-yi) ; ys <= std::min(+W, oh -2 -yi) ; ++ys) {
    for(int xs = std::max(-W, 1-xi) ; xs <= std::min(+W, ow -2 -xi) ; ++xs) {
      
      VL::float_t dx = xi + xs - x;
      VL::float_t dy = yi + ys - y;
      VL::float_t r2 = dx*dx + dy*dy ;

      // limit to a circular window
      if(r2 >= W*W+0.5) continue ;

      VL::float_t wgt = VL::fast_expn( r2 / (2*sigmaw*sigmaw) ) ;
      VL::float_t mod = *(pt + xs*xo + ys*yo) ;
      VL::float_t ang = *(pt + xs*xo + ys*yo + 1) ;

      //      int bin = (int) floor( nbins * ang / (2*M_PI) ) ;
      int bin = (int) floor( nbins * ang / (2*M_PI) ) ;
      hist[bin] += mod * wgt ;        
    }
  }
  
  // smooth the histogram
#if defined VL_LOWE_STRICT
  // Lowe's version apparently has a little issue with orientations
  // around + or - pi, which we reproduce here for compatibility
  for (int iter = 0; iter < 6; iter++) {
    VL::float_t prev  = hist[nbins/2] ;
    for (int i = nbins/2-1; i >= -nbins/2 ; --i) {
      int const j  = (i     + nbins) % nbins ;
      int const jp = (i - 1 + nbins) % nbins ;
      VL::float_t newh = (prev + hist[j] + hist[jp]) / 3.0;
      prev = hist[j] ;
      hist[j] = newh ;
    }
  }
#else
  // this is slightly more correct
  for (int iter = 0; iter < 6; iter++) {
    VL::float_t prev  = hist[nbins-1] ;
    VL::float_t first = hist[0] ;
    int i ;
    for (i = 0; i < nbins - 1; i++) {
      VL::float_t newh = (prev + hist[i] + hist[(i+1) % nbins]) / 3.0;
      prev = hist[i] ;
      hist[i] = newh ;
    }
    hist[i] = (prev + hist[i] + first)/3.0 ;
  }
#endif
  
  // find the histogram maximum
  VL::float_t maxh = * std::max_element(hist, hist + nbins) ;

  // find peaks within 80% from max
  int nangles = 0 ;
  for(int i = 0 ; i < nbins ; ++i) {
    VL::float_t h0 = hist [i] ;
    VL::float_t hm = hist [(i-1+nbins) % nbins] ;
    VL::float_t hp = hist [(i+1+nbins) % nbins] ;
    
    // is this a peak?
    if( h0 > 0.8*maxh && h0 > hm && h0 > hp ) {
      
      // quadratic interpolation
      //      VL::float_t di = -0.5 * (hp - hm) / (hp+hm-2*h0) ; 
      VL::float_t di = -0.5 * (hp - hm) / (hp+hm-2*h0) ; 
      VL::float_t th = 2*M_PI * (i+di+0.5) / nbins ;      
      angles [ nangles++ ] = th ;
      if( nangles == 4 )
        goto enough_angles ;
    }
  }
 enough_angles:
  return nangles ;
}

// ===================================================================
//                                         computeKeypointDescriptor()
// -------------------------------------------------------------------

namespace Detail {

/** Normalizes in norm L_2 a descriptor. */
void
normalize_histogram(VL::float_t* L_begin, VL::float_t* L_end)
{
  VL::float_t* L_iter ;
  VL::float_t norm = 0.0 ;

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
    norm += (*L_iter) * (*L_iter) ;

  norm = fast_sqrt(norm) ;

  for(L_iter = L_begin; L_iter != L_end ; ++L_iter)
    *L_iter /= (norm + std::numeric_limits<VL::float_t>::epsilon() ) ;
}

}

/** @brief SIFT descriptor
 **
 ** The function computes the descriptor of the keypoint @a keypoint.
 ** The function fills the buffer @a descr_pt which must be large
 ** enough. The funciton uses @a angle0 as rotation of the keypoint.
 ** By calling the function multiple times, different orientations can
 ** be evaluated.
 **
 ** @remark The function needs to compute the gradient modululs and
 ** orientation of the Gaussian scale space octave to which the
 ** keypoint belongs. The result is cached, but discarded if different
 ** octaves are visited. Thereofre it is much quicker to evaluate the
 ** keypoints in their natural octave order.
 **
 ** The function silently abort the computations of keypoints without
 ** the scale space boundaries. See also siftComputeOrientations().
 **/
void
Sift::computeKeypointDescriptor
(VL::float_t* descr_pt,
 Keypoint keypoint, 
 VL::float_t angle0)
{

  /* The SIFT descriptor is a  three dimensional histogram of the position
   * and orientation of the gradient.  There are NBP bins for each spatial
   * dimesions and NBO  bins for the orientation dimesion,  for a total of
   * NBP x NBP x NBO bins.
   *
   * The support  of each  spatial bin  has an extension  of SBP  = 3sigma
   * pixels, where sigma is the scale  of the keypoint.  Thus all the bins
   * together have a  support SBP x NBP pixels wide  . Since weighting and
   * interpolation of  pixel is used, another  half bin is  needed at both
   * ends of  the extension. Therefore, we  need a square window  of SBP x
   * (NBP + 1) pixels. Finally, since the patch can be arbitrarly rotated,
   * we need to consider  a window 2W += sqrt(2) x SBP  x (NBP + 1) pixels
   * wide.
   */      

  // octave
  int o = keypoint.o ;
  VL::float_t xperiod = getOctaveSamplingPeriod(o) ;

  // offsets to move in Gaussian scale space octave
  const int ow = getOctaveWidth(o) ;
  const int oh = getOctaveHeight(o) ;
  const int xo = 2 ;
  const int yo = xo * ow ;
  const int so = yo * oh ;

  // keypoint fractional geometry
  VL::float_t x     = keypoint.x / xperiod;
  VL::float_t y     = keypoint.y / xperiod ;
  VL::float_t sigma = keypoint.sigma / xperiod ;

  VL::float_t st0   = sinf( angle0 ) ;
  VL::float_t ct0   = cosf( angle0 ) ;
  
  // shall we use keypoints.ix,iy,is here?
  int xi = ((int) (x+0.5)) ; 
  int yi = ((int) (y+0.5)) ;
  int si = keypoint.is ;

  // const VL::float_t magnif = 3.0f ;
  const int NBO = 8 ;
  const int NBP = 4 ;
  const VL::float_t SBP = magnif * sigma ;
  const int   W = (int) floor (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;
  
  /* Offsets to move in the descriptor. */
  /* Use Lowe's convention. */
  const int binto = 1 ;
  const int binyo = NBO * NBP ;
  const int binxo = NBO ;
  // const int bino  = NBO * NBP * NBP ;
  
  int bin ;
  
  // check bounds
  if(o  < omin   ||
     o  >=omin+O ||
     xi < 0      || 
     xi > ow-1   || 
     yi < 0      || 
     yi > oh-1   ||
     si < smin+1 ||
     si > smax-2 )
        return ;
  
  // make sure gradient buffer is up-to-date
  prepareGrad(o) ;

  std::fill( descr_pt, descr_pt + NBO*NBP*NBP, 0 ) ;

  /* Center the scale space and the descriptor on the current keypoint. 
   * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
   */
  pixel_t const * pt = temp + xi*xo + yi*yo + (si - smin - 1)*so ;
  VL::float_t *  dpt = descr_pt + (NBP/2) * binyo + (NBP/2) * binxo ;
     
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)
      
  /*
   * Process pixels in the intersection of the image rectangle
   * (1,1)-(M-1,N-1) and the keypoint bounding box.
   */
  for(int dyi = std::max(-W, 1-yi) ; dyi <= std::min(+W, oh-2-yi) ; ++dyi) {
    for(int dxi = std::max(-W, 1-xi) ; dxi <= std::min(+W, ow-2-xi) ; ++dxi) {
      
      // retrieve 
      VL::float_t mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      VL::float_t angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      VL::float_t theta = fast_mod_2pi(-angle + angle0) ; // lowe compatible ?
      
      // fractional displacement
      VL::float_t dx = xi + dxi - x;
      VL::float_t dy = yi + dyi - y;
      
      // get the displacement normalized w.r.t. the keypoint
      // orientation and extension.
      VL::float_t nx = ( ct0 * dx + st0 * dy) / SBP ;
      VL::float_t ny = (-st0 * dx + ct0 * dy) / SBP ; 
      VL::float_t nt = NBO * theta / (2*M_PI) ;
      
      // Get the gaussian weight of the sample. The gaussian window
      // has a standard deviation equal to NBP/2. Note that dx and dy
      // are in the normalized frame, so that -NBP/2 <= dx <= NBP/2.
      VL::float_t const wsigma = NBP/2 ;
      VL::float_t win = VL::fast_expn((nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;
      
      // The sample will be distributed in 8 adjacent bins.
      // We start from the ``lower-left'' bin.
      int binx = fast_floor( nx - 0.5 ) ;
      int biny = fast_floor( ny - 0.5 ) ;
      int bint = fast_floor( nt ) ;
      VL::float_t rbinx = nx - (binx+0.5) ;
      VL::float_t rbiny = ny - (biny+0.5) ;
      VL::float_t rbint = nt - bint ;
      int dbinx ;
      int dbiny ;
      int dbint ;

      // Distribute the current sample into the 8 adjacent bins
      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {
            
            if( binx+dbinx >= -(NBP/2) &&
                binx+dbinx <   (NBP/2) &&
                biny+dbiny >= -(NBP/2) &&
                biny+dbiny <   (NBP/2) ) {
              VL::float_t weight = win 
                * mod 
                * fast_abs (1 - dbinx - rbinx)
                * fast_abs (1 - dbiny - rbiny)
                * fast_abs (1 - dbint - rbint) ;
              
              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }            
        }
      }
    }  
  }

  /* Standard SIFT descriptors are normalized, truncated and normalized again */
  if( normalizeDescriptor ) {

    /* Normalize the histogram to L2 unit length. */        
    Detail::normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;
    
    /* Truncate at 0.2. */
    for(bin = 0; bin < NBO*NBP*NBP ; ++bin) {
      if (descr_pt[bin] > 0.2) descr_pt[bin] = 0.2;
    }
    
    /* Normalize again. */
    Detail::normalize_histogram(descr_pt, descr_pt + NBO*NBP*NBP) ;
  }

}

// namespace VL
}
