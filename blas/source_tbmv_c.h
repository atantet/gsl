/*
 * Author:  G. Jungman
 * RCS:     $Id$
 */

  int nounit = ( Diag == CblasNonUnit );
  size_t i;
  size_t j;
  size_t id;

  if(TransA == CblasNoTrans) {
    /* form  x := A*x */

    if(Uplo == CblasUpper) {
      for(i=0; i<N; i++) {
        BASE_TYPE temp_r = 0.0;
	BASE_TYPE temp_i = 0.0;
        for(j=i+1; j<GSL_MIN(N,i+K+1); j++) {
	  temp_r += REAL(X,incX,j)*REAL(A,1,lda * i + j) - IMAG(X,incX,j)*IMAG(A,1,lda * i + j);
	  temp_i += REAL(X,incX,j)*IMAG(A,1,lda * i + j) + IMAG(X,incX,j)*REAL(A,1,lda * i + j);
	}
	if(nounit) {
	  const BASE_TYPE ax_r = REAL(X,incX,i)*REAL(A,1,lda * i + i) - IMAG(X,incX,i)*IMAG(A,1,lda * i + i);
	  const BASE_TYPE ax_i = REAL(X,incX,i)*IMAG(A,1,lda * i + i) + IMAG(X,incX,i)*REAL(A,1,lda * i + i);	  
	  REAL(X,incX,i) = temp_r + ax_r;
	  IMAG(X,incX,i) = temp_i + ax_i;
	}
	else {
	  REAL(X,incX,i) += temp_r;
	  IMAG(X,incX,i) += temp_i;
	}
      }
    }
    else {
      for(id=0; id<N; id++) {
        BASE_TYPE temp_r = 0.0;
	BASE_TYPE temp_i = 0.0;
	const size_t j_min = ( K>N-1-id ? 0 : N-1-id-K );
	i = N-1-id;
        for(j=j_min; j<i; j++) {
	  temp_r += REAL(X,incX,j)*REAL(A,1,lda * i + j) - IMAG(X,incX,j)*IMAG(A,1,lda * i + j);
	  temp_i += REAL(X,incX,j)*IMAG(A,1,lda * i + j) + IMAG(X,incX,j)*REAL(A,1,lda * i + j);
	}
	if(nounit) {
	  const BASE_TYPE ax_r = REAL(X,incX,i)*REAL(A,1,lda * i + i) - IMAG(X,incX,i)*IMAG(A,1,lda * i + i);
	  const BASE_TYPE ax_i = REAL(X,incX,i)*IMAG(A,1,lda * i + i) + IMAG(X,incX,i)*REAL(A,1,lda * i + i);	  
	  REAL(X,incX,i) = temp_r + ax_r;
	  IMAG(X,incX,i) = temp_i + ax_i;
	}
	else {
	  REAL(X,incX,i) += temp_r;
	  IMAG(X,incX,i) += temp_i;
	}
      }
    }
  }
  else {
    /* form  x := A'*x */

    if(Uplo == CblasUpper) {
      for(id=0; id<N; id++) {
        BASE_TYPE temp_r = 0.0;
	BASE_TYPE temp_i = 0.0;
	const size_t j_min = ( K>N-1-id ? 0 : N-1-id-K );
	i = N-1-id;
        for(j=j_min; j<i; j++) {
	  temp_r += REAL(X,incX,j)*REAL(A,1,lda * j + i) - IMAG(X,incX,j)*IMAG(A,1,lda * j + i);
	  temp_i += REAL(X,incX,j)*IMAG(A,1,lda * j + i) + IMAG(X,incX,j)*REAL(A,1,lda * j + i);
        }
	if(nounit) {
	  const BASE_TYPE ax_r = REAL(X,incX,i)*REAL(A,1,lda * i + i) - IMAG(X,incX,i)*IMAG(A,1,lda * i + i);
	  const BASE_TYPE ax_i = REAL(X,incX,i)*IMAG(A,1,lda * i + i) + IMAG(X,incX,i)*REAL(A,1,lda * i + i);	  
	  REAL(X,incX,i) = temp_r + ax_r;
	  IMAG(X,incX,i) = temp_i + ax_i;
        }
	else {
	  REAL(X,incX,i) += temp_r;
	  IMAG(X,incX,i) += temp_i;
        }
      }
    }
    else {
      for(i=0; i<N; i++) {
        BASE_TYPE temp_r = 0.0;
	BASE_TYPE temp_i = 0.0;
        for(j=i+1; j<GSL_MIN(N,i+K+1); j++) {
	  temp_r += REAL(X,incX,j)*REAL(A,1,lda * j + i) - IMAG(X,incX,j)*IMAG(A,1,lda * j + i);
	  temp_i += REAL(X,incX,j)*IMAG(A,1,lda * j + i) + IMAG(X,incX,j)*REAL(A,1,lda * j + i);
	}
	if(nounit) {
	  const BASE_TYPE ax_r = REAL(X,incX,i)*REAL(A,1,lda * i + i) - IMAG(X,incX,i)*IMAG(A,1,lda * i + i);
	  const BASE_TYPE ax_i = REAL(X,incX,i)*IMAG(A,1,lda * i + i) + IMAG(X,incX,i)*REAL(A,1,lda * i + i);	  
	  REAL(X,incX,i) = temp_r + ax_r;
	  IMAG(X,incX,i) = temp_i + ax_i;
	}
	else {
	  REAL(X,incX,i) += temp_r;
	  IMAG(X,incX,i) += temp_i;
	}
      }
    }
  }
