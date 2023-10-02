use std::time::Instant;

use matix::matrix::Matrix;

fn main(){
    // let m1 = Matrix::new(vec![vec![1.04, 1.95], vec![1.44, 1.58]]);
    // let m2 = Matrix::new(vec![vec![1.39, 1.76], vec![2.00, 1.02]]);
    let m1 = Matrix::random(600, 600, 0, 5);
    let m2 = Matrix::random(600, 600, 0, 5);
    // let start = Instant::now();
    // let m3 = m1.__standart_multiply(&m2);
    // let end = start.elapsed();
    // println!("{:?}", end);
    let start = Instant::now();
    let _m3 = &m1.strassen_multiply(&m2);
    let end = start.elapsed();
    println!("{:?}", end);
}