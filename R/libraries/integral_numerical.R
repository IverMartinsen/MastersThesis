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

