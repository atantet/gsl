/* spprop.c
 * 
 * Copyright (C) 2014 Patrick Alken
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

#include <config.h>
#include <stdlib.h>

#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_errno.h>

/*
gsl_spmatrix_equal()
  Return 1 if a = b, 0 otherwise
*/

int
gsl_spmatrix_equal(const gsl_spmatrix *a, const gsl_spmatrix *b)
{
  const size_t M = a->size1;
  const size_t N = a->size2;

  if (b->size1 != M || b->size2 != N)
    {
      GSL_ERROR_VAL("matrices must have same dimensions", GSL_EBADLEN, 0);
    }
  else if (a->sptype != b->sptype)
    {
      GSL_ERROR_VAL("trying to compare different sparse matrix types", GSL_EINVAL, 0);
    }
  else
    {
      const size_t nz = a->nz;
      size_t n, outerIdx;

      if (nz != b->nz)
        return 0; /* different number of non-zero elements */

      if (GSL_SPMATRIX_ISTRIPLET(a))
        {
          /*
           * triplet formats could be out of order but identical, so use
           * gsl_spmatrix_get() on b for each aij
           */
          for (n = 0; n < nz; ++n)
            {
              double bij = gsl_spmatrix_get(b, a->i[n], a->p[n]);

              if (a->data[n] != bij)
                return 0;
            }
        }
      else if (GSL_SPMATRIX_ISCCS(a) || GSL_SPMATRIX_ISCRS(a))
        {
          /*
           * for compressed, both matrices should have everything
           * in the same order
           */

          /* check inner indices and data */
          for (n = 0; n < nz; ++n)
            {
              if ((a->i[n] != b->i[n]) || (a->data[n] != b->data[n]))
                return 0;
            }

          /* AT: check outer pointers */
          for (outerIdx = 0; outerIdx < a->outerSize + 1; ++outerIdx)
            {
              if (a->p[outerIdx] != b->p[outerIdx])
                return 0;
            }
        }
      else
        {
          GSL_ERROR_VAL("unknown sparse matrix type", GSL_EINVAL, 0);
        }

      return 1;
    }
} /* gsl_spmatrix_equal() */

/** 
 * \brief Perform an element-wise greater than test on an compressed matrix.
 *
 * Perform an element-wise greater than test on an compressed matrix.
 * Attention: zero elements are not tested.
 * \param[in] m Compressed matrix.
 * \param[in] d Scalar against which to compare.
 * \return    triplet matrix resulting from the test.
 */
gsl_spmatrix *
gsl_spmatrix_gt_elements(const gsl_spmatrix *m, const double d)
{
  gsl_spmatrix *comp;
  size_t outerIdx, p, n;

  /** Allocate as triplet matrix */
  comp = gsl_spmatrix_alloc_nzmax(m->size1, m->size2, m->nz, GSL_SPMATRIX_TRIPLET);

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  gsl_spmatrix_set(comp, m->i[n], m->p[n], (double) (m->data[n] > d), 0);
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, m->i[p], outerIdx, (double) (m->data[p] > d), 0);
	    }
	}
    }
  else if(GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, outerIdx, m->i[p], (double) (m->data[p] > d), 0);
	    }
	}
    }
  
  return comp;
}

/** 
 * \brief Perform an element-wise greater or equal than test on an compressed matrix.
 *
 * Perform an element-wise greater or equal than test on an compressed matrix.
 * Attention: zero elements are not tested.
 * \param[in] m Compressed matrix.
 * \param[in] d Scalar against which to compare.
 * \return    triplet matrix resulting from the test.
 */
gsl_spmatrix *
gsl_spmatrix_ge_elements(const gsl_spmatrix *m, const double d)
{
  gsl_spmatrix *comp;
  size_t outerIdx, p, n;

  /** Allocate as triplet matrix */
  comp = gsl_spmatrix_alloc_nzmax(m->size1, m->size2, m->nz, GSL_SPMATRIX_TRIPLET);

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  gsl_spmatrix_set(comp, m->i[n], m->p[n], (double) (m->data[n] >= d), 0);
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, m->i[p], outerIdx, (double) (m->data[p] >= d), 0);
	    }
	}
    }
  else if(GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, outerIdx, m->i[p], (double) (m->data[p] >= d), 0);
	    }
	}
    }
  
  return comp;
}

/** 
 * \brief Perform an element-wise lower than test on an compressed matrix.
 *
 * Perform an element-wise lower than test on an compressed matrix.
 * Attention: zero elements are not tested.
 * \param[in] m Compressed matrix.
 * \param[in] d Scalar against which to compare.
 * \return    triplet matrix resulting from the test.
 */
gsl_spmatrix *
gsl_spmatrix_lt_elements(const gsl_spmatrix *m, const double d)
{
  gsl_spmatrix *comp;
  size_t outerIdx, p, n;

  /** Allocate as triplet matrix */
  comp = gsl_spmatrix_alloc_nzmax(m->size1, m->size2, m->nz, GSL_SPMATRIX_TRIPLET);

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  gsl_spmatrix_set(comp, m->i[n], m->p[n], (double) (m->data[n] < d), 0);
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, m->i[p], outerIdx, (double) (m->data[p] < d), 0);
	    }
	}
    }
  else if(GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, outerIdx, m->i[p], (double) (m->data[p] < d), 0);
	    }
	}
    }
  
  return comp;
}

/** 
 * \brief Perform an element-wise lower or equal than test on an compressed matrix.
 *
 * Perform an element-wise lower or equal than test on an compressed matrix.
 * Attention: zero elements are not tested.
 * \param[in] m Compressed matrix.
 * \param[in] d Scalar against which to compare.
 * \return    triplet matrix resulting from the test.
 */
gsl_spmatrix *
gsl_spmatrix_le_elements(const gsl_spmatrix *m, const double d)
{
  gsl_spmatrix *comp;
  size_t outerIdx, p, n;

  /** Allocate as triplet matrix */
  comp = gsl_spmatrix_alloc_nzmax(m->size1, m->size2, m->nz, GSL_SPMATRIX_TRIPLET);

  if (GSL_SPMATRIX_ISTRIPLET(m))
    {
      for (n = 0; n < m->nz; n++)
	{
	  gsl_spmatrix_set(comp, m->i[n], m->p[n], (double) (m->data[n] <= d), 0);
	}
    }
  else if (GSL_SPMATRIX_ISCCS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, m->i[p], outerIdx, (double) (m->data[p] <= d), 0);
	    }
	}
    }
  else if(GSL_SPMATRIX_ISCRS(m))
    {
      for (outerIdx = 0; outerIdx < m->outerSize; outerIdx++)
	{
	  for (p = m->p[outerIdx]; p < m->p[outerIdx + 1]; ++p)
	    {
	      gsl_spmatrix_set(comp, outerIdx, m->i[p], (double) (m->data[p] <= d), 0);
	    }
	}
    }
  
  return comp;
}

/** 
 * \brief Return 1 if any element is non-zero, 0 otherwise.
 *
 * Return 1 if any element is  non-zero, 0 otherwise.
 * \param[in] m Compressed matrix.
 * \return      1 if any element is non-zero, 0 otherwise.
 */
int
gsl_spmatrix_any(const gsl_spmatrix *m)
{
  size_t n;
  
  for (n = 0; n < m->nz; ++n)
    {
      if (m->data[n])
	{
	  return 1;
	}
    }
  
  return 0;
}

