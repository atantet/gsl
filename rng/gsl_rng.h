#ifndef GSL_RNG_H
#define GSL_RNG_H
#include <stdlib.h>
#include <gsl_errno.h>

typedef struct
  {
    const char *name;
    unsigned long int max;
    unsigned long int min;
    size_t size;
    void (*set) (void *state, unsigned long int seed);
    unsigned long int (*get) (void *state);
    double (*get_double) (void *state);
  }
gsl_rng_type;

typedef struct
  {
    const gsl_rng_type * type;
    void *state;
  }
gsl_rng;


/* These structs also need to appear in default.c so you can select
   them via the environment variable GSL_RNG_TYPE */

extern const gsl_rng_type *gsl_rng_cmrg;
extern const gsl_rng_type *gsl_rng_gfsr4;
extern const gsl_rng_type *gsl_rng_minstd;
extern const gsl_rng_type *gsl_rng_mrg;
extern const gsl_rng_type *gsl_rng_mt19937;
extern const gsl_rng_type *gsl_rng_r250;
extern const gsl_rng_type *gsl_rng_ran0;
extern const gsl_rng_type *gsl_rng_ran1;
extern const gsl_rng_type *gsl_rng_ran2;
extern const gsl_rng_type *gsl_rng_ran3;
extern const gsl_rng_type *gsl_rng_rand48;
extern const gsl_rng_type *gsl_rng_rand;
extern const gsl_rng_type *gsl_rng_random128_bsd;
extern const gsl_rng_type *gsl_rng_random128_glibc2;
extern const gsl_rng_type *gsl_rng_random128_libc5;
extern const gsl_rng_type *gsl_rng_random256_bsd;
extern const gsl_rng_type *gsl_rng_random256_glibc2;
extern const gsl_rng_type *gsl_rng_random256_libc5;
extern const gsl_rng_type *gsl_rng_random32_bsd;
extern const gsl_rng_type *gsl_rng_random32_glibc2;
extern const gsl_rng_type *gsl_rng_random32_libc5;
extern const gsl_rng_type *gsl_rng_random64_bsd;
extern const gsl_rng_type *gsl_rng_random64_glibc2;
extern const gsl_rng_type *gsl_rng_random64_libc5;
extern const gsl_rng_type *gsl_rng_random8_bsd;
extern const gsl_rng_type *gsl_rng_random8_glibc2;
extern const gsl_rng_type *gsl_rng_random8_libc5;
extern const gsl_rng_type *gsl_rng_random_bsd;
extern const gsl_rng_type *gsl_rng_random_glibc2;
extern const gsl_rng_type *gsl_rng_random_libc5;
extern const gsl_rng_type *gsl_rng_randu;
extern const gsl_rng_type *gsl_rng_ranf;
extern const gsl_rng_type *gsl_rng_ranlux389;
extern const gsl_rng_type *gsl_rng_ranlux;
extern const gsl_rng_type *gsl_rng_ranlxd1;
extern const gsl_rng_type *gsl_rng_ranlxd2;
extern const gsl_rng_type *gsl_rng_ranlxs0;
extern const gsl_rng_type *gsl_rng_ranlxs1;
extern const gsl_rng_type *gsl_rng_ranlxs2;
extern const gsl_rng_type *gsl_rng_ranmar;
extern const gsl_rng_type *gsl_rng_slatec;
extern const gsl_rng_type *gsl_rng_taus;
extern const gsl_rng_type *gsl_rng_transputer;
extern const gsl_rng_type *gsl_rng_tt800;
extern const gsl_rng_type *gsl_rng_uni32;
extern const gsl_rng_type *gsl_rng_uni;
extern const gsl_rng_type *gsl_rng_vax;
extern const gsl_rng_type *gsl_rng_zuf;

extern const gsl_rng_type *gsl_rng_default;
extern unsigned long int gsl_rng_default_seed;

gsl_rng *gsl_rng_alloc (const gsl_rng_type * T);
gsl_rng *gsl_rng_cpy (gsl_rng * dest, const gsl_rng * src);
gsl_rng *gsl_rng_clone (const gsl_rng * r);

void gsl_rng_free (gsl_rng * r);

void gsl_rng_set (const gsl_rng * r, unsigned long int seed);
unsigned long int gsl_rng_max (const gsl_rng * r);
unsigned long int gsl_rng_min (const gsl_rng * r);
const char *gsl_rng_name (const gsl_rng * r);
size_t gsl_rng_size (const gsl_rng * r);
void * gsl_rng_state (const gsl_rng * r);

void gsl_rng_print_state (const gsl_rng * r);

const gsl_rng_type * gsl_rng_env_setup (void);

unsigned long int gsl_rng_get (const gsl_rng * r);
double gsl_rng_uniform (const gsl_rng * r);
double gsl_rng_uniform_pos (const gsl_rng * r);
unsigned long int gsl_rng_uniform_int (const gsl_rng * r, unsigned long int n);


#ifdef HAVE_INLINE
extern inline unsigned long int gsl_rng_get (const gsl_rng * r);

extern inline unsigned long int
gsl_rng_get (const gsl_rng * r)
{
  return (r->type->get) (r->state);
}

extern inline double gsl_rng_uniform (const gsl_rng * r);

extern inline double
gsl_rng_uniform (const gsl_rng * r)
{
  return (r->type->get_double) (r->state);
}

extern inline double gsl_rng_uniform_pos (const gsl_rng * r);

extern inline double
gsl_rng_uniform_pos (const gsl_rng * r)
{
  double x ;
  do
    {
      x = (r->type->get_double) (r->state) ;
    }
  while (x == 0) ;

  return x ;
}

extern inline unsigned long int gsl_rng_uniform_int (const gsl_rng * r, unsigned long int n);

extern inline unsigned long int
gsl_rng_uniform_int (const gsl_rng * r, unsigned long int n)
{
  unsigned long int offset = r->type->min;
  unsigned long int range = r->type->max - offset;
  unsigned long int scale = range / n;
  unsigned long int k;

  if (n > range) 
    {
      GSL_ERROR_RETURN ("n exceeds maximum value of generator",
			GSL_EINVAL, 0) ;
    }

  do
    {
      k = (((r->type->get) (r->state)) - offset) / scale;
    }
  while (k >= n);

  return k;
}
#endif /* HAVE_INLINE */

#endif /* GSL_RNG_H */
