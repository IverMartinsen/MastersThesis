make_contrast = function(n, difference = TRUE){
  #' function for making contrast matrix for all possible
  #' differences between n variables
  #' 
  #' if difference == FALSE the result will be
  #' pairwise sums
  #' 
  #' returns (n choose 2) x n matrix
  B = matrix(0, nrow = n * (n - 1) / 2, ncol = n)
  
  if(difference){
    number = -1
  } else {
    number = 1
  }
  
  counter = 0
  for(j in (n - 1):1){
    for(i in 1:j){
      B[i + counter, n - j] = 1
      B[i + counter, n - j + i] = number
    }
    counter = counter + j
  }
  return(B)
}
