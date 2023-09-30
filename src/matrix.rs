use std::ops::{Index, Add, AddAssign};
pub struct Matrix{
    pub rows:usize,
    pub cols:usize,
    pub data:Vec<f32>
}

impl Matrix{
    pub fn new<T: IntoMatrix>(data: T) -> Self{
        data.into_matrix()
    }
    fn elementwise_op_in_place<F>(&mut self, other: &Matrix, op:F)
    where F:Fn(f32,f32) ->f32,
    {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        
        for (a, &b) in self.data.iter_mut().zip(&other.data) {
            *a = op(*a, b);
        }
    }
    fn elementwise_op_new<F> (&self, other: &Matrix, op:F) -> Matrix
    where F:Fn(f32, f32) -> f32,
    {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut data: Vec<f32> = vec![0.0; self.data.len()];
    
        for ((res, a), b) in data.iter_mut().zip(&self.data).zip(&other.data) {
            *res = op(*a, *b);
        }
    
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: data,
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
        let cols = if rows > 0 {self[0].len()} else {0};
        let data = self.into_iter().flatten().collect::<Vec<_>>();
        Matrix{
            rows,
            cols,
            data
        }
    }
}

impl Add for Matrix{
    type Output = Matrix;

    fn add(self, other: Matrix) -> Matrix{
        self.elementwise_op_new(&other, |a,b| a+b)
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        self.elementwise_op_in_place(&rhs, |a,b| a + b)
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

