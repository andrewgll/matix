use core::fmt;
use std::{ops::{Index, Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Range, RangeFrom}, cmp::max, arch::asm};
use crossbeam_channel::bounded;
use packed_simd::f32x16;
use rand::Rng;
use rayon::prelude::*;
#[derive(Clone)]
#[derive(Debug)]
pub struct Matrix{
    pub rows:usize,
    pub cols:usize,
    pub data:Vec<f32>
}

impl Matrix{
    pub fn new<T: IntoMatrix>(data: T) -> Self{
        data.into_matrix()
    }
    pub fn random(rows: usize, cols: usize, min:f32, max:f32) -> Self{
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.gen_range(min..max))
            .collect();
        Matrix { rows: rows, cols: cols, data: data }
    }
    pub fn __strassen_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);

        // Base case
        if max(self.rows, self.cols) <= 64 {
            return 
            Matrix {
                cols: self.cols,
                rows: self.rows,
                data: self.__rayon_simd_multiply(&other)
            };
        }

        let (a11, a12, a21, a22) = self.split();
        let (b11, b12, b21, b22) = other.split();

        let (sender, receiver) = bounded(7); 
        rayon::scope(|s| {
            s.spawn(|_| {
                sender.send(a11.__strassen_multiply(&(&b12 - &b22))).unwrap();
            });
            s.spawn(|_| {
                sender.send((&a11 + &a12).__strassen_multiply(&b22)).unwrap();
            });
            s.spawn( |_| {
                sender.send((&a21 + &a22).__strassen_multiply(&b11)).unwrap(); 
            });
            s.spawn( |_| {
                sender.send(a22.__strassen_multiply(&(&b21 - &b11))).unwrap(); 
            });
            s.spawn(|_| {
                sender.send((&a11 + &a22).__strassen_multiply(&(&b11 + &b22))).unwrap(); 
            });
            s.spawn(|_| {
                sender.send((&a12 - &a22).__strassen_multiply(&(&b21 + &b22))).unwrap(); 
            });
            s.spawn(|_| {
                sender.send((&a11 - &a21).__strassen_multiply(&(&b11 + &b12))).unwrap();
            });
        });
        let results: Vec<Matrix> = receiver.iter().take(7).collect();
        let c11 = &(&(&results[4] + &results[3]) - &results[1]) + &results[5];
        let c12 = &results[0] + &results[1];
        let c21 = &results[2] + &results[3];
        let c22 = &(&(&results[4] + &results[0]) - &results[2]) - &results[6];
    
        // Merge the results
        Matrix::merge(c11, c12, c21, c22)
    }
    fn split(&self) -> (Matrix, Matrix, Matrix, Matrix) {
        let mid = self.rows / 2;
        
        let mut a11_data = Vec::with_capacity(mid * mid);
        let mut a12_data = Vec::with_capacity(mid * mid);
        let mut a21_data = Vec::with_capacity(mid * mid);
        let mut a22_data = Vec::with_capacity(mid * mid);
    
        for i in 0..self.rows {
            if i < mid {
                a11_data.extend_from_slice(&self.data[i * self.cols..i * self.cols + mid]);
                a12_data.extend_from_slice(&self.data[i * self.cols + mid..(i + 1) * self.cols]);
            } else {
                a21_data.extend_from_slice(&self.data[i * self.cols..i * self.cols + mid]);
                a22_data.extend_from_slice(&self.data[i * self.cols + mid..(i + 1) * self.cols]);
            }
        }
    
        (
            Matrix { rows: mid, cols: mid, data: a11_data },
            Matrix { rows: mid, cols: mid, data: a12_data },
            Matrix { rows: mid, cols: mid, data: a21_data },
            Matrix { rows: mid, cols: mid, data: a22_data }
        )
    }
    fn merge(a11: Matrix, a12: Matrix, a21: Matrix, a22: Matrix) -> Matrix {
        let new_rows = a11.rows + a21.rows;
        let new_cols = a11.cols + a12.cols;
    
        let mut data = Vec::with_capacity(new_rows * new_cols);
    
        data.extend(&a11.data);
        data.extend(&a12.data);
        data.extend(&a21.data);
        data.extend(&a22.data);
    
        Matrix { rows: new_rows, cols: new_cols, data: data }
    }
    
    pub fn __rayon_simd_multiply(&self, other: &Matrix) -> Vec<f32> {
        assert_eq!(self.cols, other.rows, "Matrix dimensions do not match for multiplication");
    
        let mut data = vec![0.0; self.rows * other.cols];
        let self_cols = self.cols;
        let other_cols = other.cols;
        let lanes = f32x16::lanes();
        let chunks = other_cols / lanes;
    
        data.par_chunks_mut(other_cols).enumerate().for_each(|(i, row)| {
            for k in 0..self_cols {
                let a_ik = self.data[i * self_cols + k];
                let simd_a = f32x16::splat(a_ik);
                for j in (0..chunks).map(|x| x * lanes) {
                    let simd_b = f32x16::from_slice_unaligned(&other.data[k * other_cols + j..]);
                    let simd_res = f32x16::from_slice_unaligned(&row[j..]);
                    (simd_res + simd_a * simd_b).write_to_slice_unaligned(&mut row[j..]);
    
                    // Prefetching for next iteration
                    unsafe {
                        let next_ptr: *const f32 = &other.data[k * other_cols + j];
                        asm!(
                            "prefetcht0 [{0}]",
                            in(reg) next_ptr,
                            options(nostack, preserves_flags)
                        );
                    }
                }
                // Loop unrolling for the non-SIMD portion
                let mut j = chunks * lanes;
                while j < other_cols {
                    row[j] += a_ik * other.data[k * other_cols + j];
                    if j + 1 < other_cols {
                        row[j + 1] += a_ik * other.data[k * other_cols + j + 1];
                    }
                    if j + 2 < other_cols {
                        row[j + 2] += a_ik * other.data[k * other_cols + j + 2];
                    }
                    if j + 3 < other_cols {
                        row[j + 3] += a_ik * other.data[k * other_cols + j + 3];
                    }
                    j += 4;
                }
            }
        });
        data
    }
    
    
    fn elementwise_op_in_place<F>(&mut self, other: Option<&Matrix>, scalar: f32, op:F)
    where F:Fn(f32,f32) ->f32,
    {
        match other {
            Some(matrix) => {
                
                assert_eq!(self.rows, matrix.rows);
                assert_eq!(self.cols, matrix.cols);
                
                for (a, &b) in self.data.iter_mut().zip(&matrix.data) {
                    *a = op(*a, b);
                }
            },
            None => {
                for a in &mut self.data{
                    *a = op(*a, scalar);
                }
            }
        }
    }
    fn elementwise_op_new<F>(&self, other: Option<&Matrix>, scalar: f32, op: F) -> Matrix
    where
        F: Fn(f32, f32) -> f32,
    {
        let data = match other {
        Some(matrix) => {
            assert_eq!(self.rows, matrix.rows);
            assert_eq!(self.cols, matrix.cols);

            self.data.iter()
                     .zip(matrix.data.iter())
                     .map(|(a, b)| op(*a, *b))
                     .collect::<Vec<f32>>()
        }
        None => {
            self.data.iter()
                     .map(|&a| op(a, scalar))
                     .collect::<Vec<f32>>()
        }
    };

    Matrix {
        rows: self.rows,
        cols: self.cols,
        data,
    }
}
}

pub trait IntoMatrix{
    fn into_matrix(self) -> Matrix;
}

impl IntoMatrix for Vec<f32>{
    fn into_matrix(self) -> Matrix{
        Matrix{
            rows: 1,
            cols: self.len(),
            data: self,
        }
    }
}

impl IntoMatrix for Vec<Vec<f32>>{
    fn into_matrix(self) -> Matrix{
        let rows = self.len();
        let cols = self.get(0).map_or(0, |r| r.len());

        if !self.iter().all(|r| r.len() == cols){
            panic!("All rows must have the same number of columns.");
        }
        let data = self.into_iter().flatten().collect::<Vec<_>>();
        Matrix{
            rows,
            cols,
            data
        }
    }
}

impl fmt::Display for Matrix{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        write!(f, "([\n")?;
        for i in 0..self.rows {
            write!(f,"  [")?;
            for j in 0..self.cols {
                if j < self.cols - 1 {
                    write!(f, "{:6.2}, ",  self.data[i*self.cols+j])?;
                }
                else{
                    write!(f, "{:6.2}",  self.data[i*self.cols+j])?;
                }
            }
            if i < self.rows - 1{
                writeln!(f, "],")?;
            }
            else{
                writeln!(f,"]")?;
            }
        }
        write!(f,"]);\n")?;
        Ok(())
    }
}

impl<'a,'b> Add<&'b Matrix> for &'a Matrix{
    type Output = Matrix;

    fn add(self, other: &'b Matrix) -> Matrix{
        self.elementwise_op_new(Some(&other),0.0, |a,b| a+b)
    }
}

impl Add for Matrix {
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix{
        &self + &other
    }
}

impl Add<f32> for Matrix{
    type Output = Matrix;

    fn add(self, other: f32) -> Matrix{
        self.elementwise_op_new(None, other, |a,b| a+b)
    }
}

impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, other: &'b Matrix) -> Matrix{
        self.elementwise_op_new(Some(&other),0.0, |a,b| a-b)
    }
}

impl Sub for Matrix {
    type Output = Matrix;

    fn sub(self, other: Matrix) -> Matrix{
        &self - &other
    }
}

impl Sub<f32> for Matrix {
    type Output = Matrix;

    fn sub(self, other: f32) -> Matrix{
        self.elementwise_op_new(None,other, |a,b| a-b)
    }
}

impl Mul<f32> for Matrix{
    type Output = Matrix;

    fn mul(self, other: f32) -> Matrix{
        self.elementwise_op_new(None, other, |a,b| a*b)
    }
}

impl Mul for Matrix{
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix{
        self.__strassen_multiply(&other)
    }
}

impl<'a,'b> Mul<&'b Matrix> for &'a Matrix{
    type Output = Matrix;

    fn mul(self, other: &'b Matrix) -> Matrix{
        self.__strassen_multiply(&other)
    }
}

impl MulAssign<f32> for Matrix{
    fn mul_assign(&mut self, rhs: f32) {
        self.elementwise_op_in_place(None, rhs, |a,b| a*b)
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Matrix) {
        self.elementwise_op_in_place(Some(&rhs),0.0, |a,b| a-b)
    }
}

impl SubAssign<f32> for Matrix {
    fn sub_assign(&mut self, rhs:f32) {
        self.elementwise_op_in_place(None, rhs, |a,b| a-b)
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Matrix) {
        self.elementwise_op_in_place(Some(&rhs),0.0, |a,b| a + b)
    }
}

impl AddAssign<f32> for Matrix {
    fn add_assign(&mut self, rhs: f32) {
        self.elementwise_op_in_place(None, rhs, |a,b| a + b)
    }
}

impl Index<usize> for Matrix {
    type Output = [f32];
    fn index(&self, row: usize) -> &Self::Output {
        let start = row * self.cols;
        let end = start + self.cols;
        &self.data[start..end]
    }
}
impl Index<Range<usize>> for Matrix {
    type Output = [f32];
    
    fn index(&self, rows: Range<usize>) -> &Self::Output {
        let start = rows.start * self.cols;
        let end = rows.end * self.cols;
        &self.data[start..end]
    }
}

impl Index<RangeFrom<usize>> for Matrix {
    type Output = [f32];
    
    fn index(&self, rows: RangeFrom<usize>) -> &Self::Output {
        let start = rows.start * self.cols;
        &self.data[start..]
    }
}