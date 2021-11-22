# Utility functions


integrate_num = function(x, a, b, method){
  # Numerical integration using the trapezoid or the simpsons rule
  m = length(x) - 1
  h = (b - a) / m
  if(method == 'trapezoid'){
    return(sum((x[1:m-1] + x[2:m])*h/2))    
  }
  else if(method == 'simpson'){
    return((x[1] + x[m+1] + 4*sum(x[seq(2, m, 2)]) + 2*sum(x[seq(3, m-1, 2)]))*h/3)    
  }
}


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
