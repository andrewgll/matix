use matix::matrix::Matrix;

fn main(){
    let m = Matrix::new(vec![1.2,1.3]);
    let m2 = Matrix::new(vec![1.2,1.3]);
    let m3 = m+m2;
    print!("{}", m3[0][0]) 
}