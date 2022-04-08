# Utility functions


integrate_num = function(y, a, b, method){
  #' Numerical integration using the trapezoid or the simpsons rule.
  #' 
  #' y: vector of function values (numeric)
  #' a: start of interval (double)
  #' b: end of interval (double)
  #' method: rule for integration (character), either "trapezoid" or "simpson"
  #' 
  #' Return: integral (double)
  m = length(y) - 1
  h = (b - a) / m
  if(method == 'trapezoid'){
    return(sum((y[1:m-1] + y[2:m])*h/2))    
  }
  else if(method == 'simpson'){
    return((y[1] + y[m+1] + 4*sum(y[seq(2, m, 2)]) + 2*sum(y[seq(3, m-1, 2)]))*h/3)    
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


compute_kl_divergence = function(y, z){
    #' Compute Kullback-Leibler divergence of y wrt z
    #' 
    #' y: vector of function values (numeric)
    #' z: vector of function values (numeric)
    #' 
    #' Return: KL-divergence (double)
    
    # Obtain x-range for integration
    a = round(min(density(y)$x))
    b = round(max(density(y)$x))
    # Compute y density estimates for read age (p) and predicted age (q)
    p = density(z, from = a, to = b)$y
    q = density(y, from = a, to = b)$y
    # Return KL-divergence
    return(-integrate_num(p*log(q/p), a, b, 'simpson'))  
}

