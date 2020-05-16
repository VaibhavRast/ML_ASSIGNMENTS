SVM algorithm was applied to determine whether a given mail was spam or not. The dataset consisted of 4601 rows and 58 columns. 
Different kernel functions were applied(linear,RBF and quadratic) with different values of C and gamma was also varied for RBF kernel function.
Linear kernel gave a test accuracy of 93.26%,quadratic kernel gave a test accuracy of 88.3% and RBF kernel gave a test accuracy of 94.67%.
Linear kernel gave a training accuracy of 93.28%,quadratic kernel gave a training accuracy of 87.34% and RBF kernel gave a training accuracy of 94.78%. 

Explanation:-

1) For linear kernel function, optimal C value was found out to be 1.At this C value, a test accuracy of 93.26% was obtained.
2) For quadratic kernel function, test accuracy kept increasing with increase in the value of C.At C=100000, a test accuracy of 89.9% was obtained.
   However, training was stopped at C=100000 since increasing the C value further could have resulting in slight overfitting and not doing well for the new data and penalizing the misclassified points heavily.
   
3) For RBF kernel function, test accuracy kept increasing with increase in the value of C. At c=100000, an accuracy of 94.67% was obtained.
   However, training was stopped at C=100000 since increasing the C value further could have resulting in slight overfitting and not doing well for the new data and penalizing the misclassified points heavily.
   

  On applying all the 3 kernel functions, it was found that RBF kernel gave the best accuracy.
  However, linear kernel function suffers least from the problem of overfitting since it gave an optimal accuracy value at C=1 itself and it gave a good accuracy of 93.26%.
