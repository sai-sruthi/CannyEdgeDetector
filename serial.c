#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <float.h>
#include <string.h>
#include <stdbool.h>



#define MAX_BRIGHTNESS 255
 double start , end;//for the calculation of time.
 
typedef struct {
    char check[2];
} bmp_check;

typedef struct {
    unsigned int files;
    unsigned short creator1;
    unsigned short creator2;
    unsigned int bmp_offset;
} bmp_header;
 
typedef struct {
    unsigned int header;
    int  width;
    int  height;
    unsigned short nplanes;
    unsigned short bitspp;
    unsigned int compress_type;
    unsigned int bmp_bytes;
    int  hres;
    int  vres;
    unsigned int ncolors;
    unsigned int nimpcolors;
} bmp_infoheader;
 
typedef struct {
    char r;
    char g;
    char b;
    char extra;
} rgb;

// Use short int instead `unsigned char' so that we can
// store negative values.
typedef short int pixel;
 
pixel *load_bmp(const char *filename,bmp_infoheader *bitmapInfoHeader)
{
    FILE *filePtr = fopen(filename, "rb");
    if (filePtr == NULL) {
        perror("fopen()");
        return NULL;
    }

    
    bmp_check ch;
    if (fread(&ch, sizeof(bmp_check), 1, filePtr) != 1) {
        fclose(filePtr);
        return NULL;
    }
 
    // verify that this is a bmp file by check bitmap id

    if (*((unsigned short*)ch.check) != 0x4D42) {
        fprintf(stderr, "Not a BMP file: check=%c%c\n",
                ch.check[0], ch.check[1]);
        fclose(filePtr);
        return NULL;
    }
    
    
    bmp_header bitmapFileHeader; // our bitmap file header
    // read the bitmap file header
    if (fread(&bitmapFileHeader, sizeof(bmp_header),1, filePtr) != 1) {
        fclose(filePtr);
        return NULL;
    }
 
    // read the bitmap info header
    if (fread(bitmapInfoHeader, sizeof(bmp_infoheader),1, filePtr) != 1) {
        fclose(filePtr);
        return NULL;
    }
 
    //compression not allowed
    if (bitmapInfoHeader->compress_type != 0)
        fprintf(stderr, "Warning, compression is not supported.\n");
 
    // move file point to the beginning of bitmap data
    if (fseek(filePtr, bitmapFileHeader.bmp_offset, SEEK_SET)) {
        fclose(filePtr);
        return NULL;
    }
 
    // allocate enough memory for the bitmap image data
    pixel *bitmapimage = malloc(bitmapInfoHeader->bmp_bytes *sizeof(pixel));
 
    // verify memory allocation
    if (bitmapimage == NULL) {
        fclose(filePtr);
        return NULL;
    }
 
    // read in the bitmap image data
    size_t pad, count=0;
    unsigned char c;
    pad = 4*ceil(bitmapInfoHeader->bitspp*bitmapInfoHeader->width/32.) - bitmapInfoHeader->width;
    size_t i,j;
    for(i=0; i<bitmapInfoHeader->height; i++){
	    for(j=0; j<bitmapInfoHeader->width; j++){
		    if (fread(&c, sizeof(unsigned char), 1, filePtr) != 1) {
			    fclose(filePtr);
			    return NULL;
		    }
		    bitmapimage[count++] = (pixel) c;
	    }
	    fseek(filePtr, pad, SEEK_CUR);//according to bitmap image format
    }
     
    
    fclose(filePtr);
    return bitmapimage;
}
 
 
// Return: true on error.
bool save_bmp(const char *filename, const bmp_infoheader *bmp_ih,const pixel *data)
{
    FILE* filePtr = fopen(filename, "wb");
    if (filePtr == NULL)
        return true;
 
    bmp_check ch = {{0x42, 0x4d}};
    if (fwrite(&ch, sizeof(bmp_check), 1, filePtr) != 1) {
        fclose(filePtr);
        return true;
    }
 
    const unsigned int offset = sizeof(bmp_check) +
                            sizeof(bmp_header) +
                            sizeof(bmp_infoheader) +
                            ((1U << bmp_ih->bitspp) * 4);
 
    const bmp_header bmp_fh = {
        .files = offset + bmp_ih->bmp_bytes,
        .creator1 = 0,
        .creator2 = 0,
        .bmp_offset = offset
    };
 
    if (fwrite(&bmp_fh, sizeof(bmp_header), 1, filePtr) != 1) {
        fclose(filePtr);
        return true;
    }
    if (fwrite(bmp_ih, sizeof(bmp_infoheader), 1, filePtr) != 1) {
        fclose(filePtr);
        return true;
    }
 
    //creating rgb pixels so as to write to the file
    size_t i,j;
   for (i = 0; i < (1U << bmp_ih->bitspp); i++) {
        const rgb color = {(char)i, (char)i, (char)i};
        if (fwrite(&color, sizeof(rgb), 1, filePtr) != 1) {
            fclose(filePtr);
            return true;
        }
    }
    

    size_t pad = 4*ceil(bmp_ih->bitspp*bmp_ih->width/32.) - bmp_ih->width;
    unsigned char c;
    for(i=0; i < bmp_ih->height; i++) {
	    for(j=0; j < bmp_ih->width; j++) {
		    c = (unsigned char) data[j + bmp_ih->width*i];
		    if (fwrite(&c, sizeof(char), 1, filePtr) != 1) {
			    fclose(filePtr);
			    return true;
		    }
	    }
	    c = 0;
	    for(j=0; j<pad; j++)
		    if (fwrite(&c, sizeof(char), 1, filePtr) != 1) {
			    fclose(filePtr);
			    return true;
		    }
    }
 
    fclose(filePtr);
    return false;
}
 
 
 
// if mid is true, map pixels to range 0..MAX_BRIGHTNESS
void convolution(const pixel *in, pixel *out, const float *kernel,
                 const int nx, const int ny, const int kn,
                 const bool mid)
{
    assert(kn % 2 == 1);
    assert(nx > kn && ny > kn);
    const int kernelhalf = kn / 2;
    float min = FLT_MAX, max = -FLT_MAX;
    int m,n,j,i;
    float pix=0.0;
    start=omp_get_wtime();
    if (mid)
    {
        for (m = kernelhalf; m < nx - kernelhalf; m++)
            for (n = kernelhalf; n < ny - kernelhalf; n++) {
                pix = 0.0;
                size_t c = 0;
                for (j = -kernelhalf; j <= kernelhalf; j++)
                    for (i = -kernelhalf; i <= kernelhalf; i++) {
                        pix += in[(n - j) * nx + m - i] * kernel[c];
                        c++;
                    }
                if (pix < min)
                    min = pix;
                if (pix > max)
                    max = pix;
                }
    }
	
        for (m = kernelhalf; m < nx - kernelhalf; m++)
        for (n = kernelhalf; n < ny - kernelhalf; n++) {
            pix = 0.0;
            size_t c = 0;
            for (j = -kernelhalf; j <= kernelhalf; j++)
                for (i = -kernelhalf; i <= kernelhalf; i++) {
                    pix += in[(n - j) * nx + m - i] * kernel[c];
                    c++;
                }
 
            if (mid)
                pix = MAX_BRIGHTNESS * (pix - min) / (max - min);
            out[n * nx + m] = (pixel)pix;
        }
        end=omp_get_wtime();
        printf("\nConvolution time: \t\t %f",end-start);
}
 

 
 
void gaussian_filter(const pixel *in, pixel *out,
                     const int nx, const int ny, const float sigma)
{
    const int n = 2 * (int)(2 * sigma) + 3;
    const float mean = (float)floor(n / 2.0);
    float kernel[n * n]; // variable length array
 
    
    fprintf(stderr, "gaussian_filter: kernel size_t %d, sigma=%g\n",
            n, sigma);
    size_t c = 0;
    int i,j;
    
    start=omp_get_wtime();
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            kernel[c] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) +
                                    pow((j - mean) / sigma, 2.0)))
                        / (2 * M_PI * sigma * sigma);
            c++;
        }
    end=omp_get_wtime();
    printf("\nGaussian filter time: \t\t %f",end-start);
    convolution(in, out, kernel, nx, ny, n, true);
    
}
 
 
 
 
 
 
pixel *canny_edge_detection(const pixel *in,
                              const bmp_infoheader *bmp_ih,
                              const int tmin, const int tmax,
                              const float sigma)
{
    const int nx = bmp_ih->width;
    const int ny = bmp_ih->height;
 
    pixel *G = calloc(nx * ny * sizeof(pixel), 1);
    pixel *after_Gx = calloc(nx * ny * sizeof(pixel), 1);
    pixel *after_Gy = calloc(nx * ny * sizeof(pixel), 1);
    pixel *nms = calloc(nx * ny * sizeof(pixel), 1);
    pixel *out = malloc(bmp_ih->bmp_bytes * sizeof(pixel));
 
    if (G == NULL || after_Gx == NULL || after_Gy == NULL ||
        nms == NULL || out == NULL) {
        fprintf(stderr, "canny_edge_detection:"
                " Failed memory allocation(s).\n");
        exit(1);
    }
 
    gaussian_filter(in, out, nx, ny, sigma);
    save_bmp("smooth.bmp",bmp_ih,out);
    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
 
    convolution(out, after_Gx, Gx, nx, ny, 3, false);
 
    const float Gy[] = { 1, 2, 1,
                         0, 0, 0,
                        -1,-2,-1};
 
    convolution(out, after_Gy, Gy, nx, ny, 3, false);
    
    
    //gradient-finding
    int i,j; 
    start=omp_get_wtime();  
    
    for (i = 1; i < nx - 1; i++)
        for (j = 1; j < ny - 1; j++) {
            const int c = i + nx * j;
            G[c] = (pixel)hypot(after_Gx[c], after_Gy[c]);
        }
  end=omp_get_wtime();
  save_bmp("gradientx.bmp",bmp_ih,after_Gx);
  save_bmp("gradienty.bmp",bmp_ih,after_Gy);
  printf("\nGradient time: \t\t\t%f\n",end-start);
  
  
 start=omp_get_wtime();
    // Non-maximum suppression implementation.

 for (i = 1; i < nx - 1; i++)
    for (j = 1; j < ny - 1; j++) {
            const int c = i + nx * j;
            const int nn = c - nx;
            const int ss = c + nx;
            const int ww = c + 1;
            const int ee = c - 1;
            const int nw = nn + 1;
            const int ne = nn - 1;
            const int sw = ss + 1;
            const int se = ss - 1;
 
            const float dir = (float)(fmod(atan2(after_Gy[c],
                                                 after_Gx[c]) + M_PI,
                                           M_PI) / M_PI) * 8;
 
            if (((dir <= 1 || dir > 7) && G[c] > G[ee] &&
                 G[c] > G[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && G[c] > G[nw] &&
                 G[c] > G[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && G[c] > G[nn] &&
                 G[c] > G[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && G[c] > G[ne] &&
                 G[c] > G[sw]))   // 135 deg
                nms[c] = G[c];
            else
                nms[c] = 0;
        }
 end=omp_get_wtime();
 save_bmp("non_max.bmp",bmp_ih,nms);
 printf("Time for Non_max:\t\t%f\n",end-start);
    
 
 
    int *edges = (int*) after_Gy;
    memset(out, 0, sizeof(pixel) * nx * ny);
    memset(edges, 0, sizeof(pixel) * nx * ny);
 
    start=omp_get_wtime();
 
    // Tracing edges with hysteresis .
    size_t c = 1;
  
    for (j = 1; j < ny - 1; j++)
        for (i = 1; i < nx - 1; i++) {
            if (nms[c] >= tmax && out[c] == 0) { // trace edges
                out[c] = MAX_BRIGHTNESS;
                int nedges = 1;
                edges[0] = c;
 
                do {
                    nedges--;
                    const int t = edges[nedges];
 
                    int nbs[8]; // neighbours
                    nbs[0] = t - nx;     // nn
                    nbs[1] = t + nx;     // ss
                    nbs[2] = t + 1;      // ww
                    nbs[3] = t - 1;      // ee
                    nbs[4] = nbs[0] + 1; // nw
                    nbs[5] = nbs[0] - 1; // ne
                    nbs[6] = nbs[1] + 1; // sw
                    nbs[7] = nbs[1] - 1; // se
		    int k;
                    for (k = 0; k < 8; k++)
                        if (nms[nbs[k]] >= tmin && out[nbs[k]] == 0) {
                            out[nbs[k]] = MAX_BRIGHTNESS;
                            edges[nedges] = nbs[k];
                            nedges++;
                        }
                } while (nedges > 0);
            }
            c++;
        }
 
 end=omp_get_wtime();
 save_bmp("hysteresis.bmp",bmp_ih,out);
 printf("Time for Hysterisis:\t\t%f\n\n\n\n",end-start);
 
    free(after_Gx);
    free(after_Gy);
    free(G);
    free(nms);
 
    return out;
}
 
int main(const int argc, const char ** const argv)
{
    if (argc < 2) {
        printf("Usage: %s image.bmp \n", argv[0]);
        return 1;
    }

    static bmp_infoheader ih;
    const pixel *in_bitmap_data = load_bmp(argv[1], &ih);
    if (in_bitmap_data == NULL) {
        fprintf(stderr, "main: BMP image not loaded.\n");
        return 1;
    }
 
    printf("Info: %d x %d x %d\n", ih.width, ih.height, ih.bitspp);
 
    const pixel *out_bitmap_data =
        canny_edge_detection(in_bitmap_data, &ih, 45, 50, 1.0f);
    if (out_bitmap_data == NULL) {
        fprintf(stderr, "main: failed canny_edge_detection.\n");
        return 1;
    }
 
    if (save_bmp("outserial.bmp", &ih, out_bitmap_data)) {
        fprintf(stderr, "main: BMP image not saved.\n");
        return 1;
    }
 
    free((pixel*)in_bitmap_data);
    free((pixel*)out_bitmap_data);
    return 0;
}