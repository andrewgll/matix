use crate::matrix::Matrix;


pub struct MemoryPool {
    pub chunks: Vec<Matrix>,
    pub free_indices: Vec<usize>,
}

impl MemoryPool {
    pub fn new(chunk_size: usize, num_chunks: usize) -> Self {
        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            chunks.push(Matrix::with_dimensions(chunk_size, chunk_size));
        }
        let free_indices = (0..num_chunks).collect::<Vec<_>>();
        MemoryPool { chunks, free_indices }
    }

    pub fn allocate(&mut self) -> Option<&mut Matrix> {
        self.free_indices.pop().map(|index| &mut self.chunks[index])
    }

    pub fn deallocate(&mut self, matrix: &mut Matrix) {
        let index = (matrix as *mut _ as usize - self.chunks.as_ptr() as usize) / std::mem::size_of::<Matrix>();
        self.free_indices.push(index);
        
    }

    pub fn add_chunks(&mut self, chunk_size: usize, num_chunks: usize) {
        let current_length = self.chunks.len();
        for _ in 0..num_chunks {
            self.chunks.push(Matrix::with_dimensions(chunk_size, chunk_size));
            self.free_indices.push(current_length);
        }
    }
}
