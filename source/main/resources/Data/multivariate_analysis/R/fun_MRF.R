####################################################################################################
######                            Multivariate Random Forest                              #######
####################################################################################################
build_forest_predict <- function(trainX, trainY, n_tree, m_feature, min_leaf, testX){
  if (class(n_tree)=="character" || n_tree%%1!=0 || n_tree<1) stop('Number of trees in the forest can not be fractional or negative integer or string')
  if (class(m_feature)=="character" || m_feature%%1!=0 || m_feature<1) stop('Number of randomly selected features considered for a split can not be fractional or negative integer or string')
  if (class(min_leaf)=="character" || min_leaf%%1!=0 || min_leaf<1 || min_leaf>nrow(trainX)) stop('Minimum leaf number can not be fractional or negative integer or string or greater than number of samples')
  
  theta <- function(trainX){trainX}
  results <- bootstrap::bootstrap(1:nrow(trainX),n_tree,theta) 
  b=results$thetastar
  
  Variable_number=ncol(trainY)
  if (Variable_number>1){
    Command=2
  }else if(Variable_number==1){
    Command=1
  } 
  
  Y_HAT=matrix(  0*(1:Variable_number*nrow(testX)),  ncol=Variable_number,   nrow=nrow(testX)  )
  Y_pred=NULL
  
  for (i in 1:n_tree){
    Single_Model=NULL
    X=trainX[ b[ ,i],  ]
    Y=matrix(trainY[ b[ ,i],  ],ncol=Variable_number)
    Inv_Cov_Y = solve(cov(Y), tol=1e-40) # calculate the V inverse
    if (Command==1){
      Inv_Cov_Y=matrix(rep(0,4),ncol=2)
    }
    Single_Model=build_single_tree(X, Y, m_feature, min_leaf,Inv_Cov_Y,Command)
    Y_pred=single_tree_prediction(Single_Model,testX,Variable_number)
    for (j in 1:Variable_number){
      Y_HAT[,j]=Y_HAT[,j]+Y_pred[,j]
    }
  }
  Y_HAT=Y_HAT/n_tree
  return(Y_HAT)
}
#######################################################
build_single_tree <- function(X, Y, m_feature, min_leaf,Inv_Cov_Y,Command){
  NN=round(nrow(X)/min_leaf)*50
  model=rep( list(NULL), NN )
  i=1
  Index=1:nrow(X)
  
  model=split_node(X,Y,m_feature,Index,i,model,min_leaf,Inv_Cov_Y,Command)
  return(model)
}
#######################################################
split_node <- function(X,Y,m_feature,Index,i,model,min_leaf,Inv_Cov_Y,Command){
  ii=NULL
  Index_left=NULL
  Index_right=NULL
  if(length(Index)>min_leaf){
    ff2 = ncol(X) # number of features
    ff =sort(sample(ff2, m_feature)) #randomly taken m_feature features, for each splits vary
    # Result = splitt(X,Y,m_feature,Index,Inv_Cov_Y,Command, ff)
    Result = MultivariateRandomForest::splitt2(X,Y,m_feature,Index,Inv_Cov_Y,Command, ff)
    Index_left=Result[[1]]
    Index_right=Result[[2]]
    if(i==1){
      Result[[5]]=c(2,3)
    }else{
      j=1
      while (length(model[[j]])!=0){
        j=j+1
      }
      Result[[5]]=c(j,j+1)
    }
    
    model[[i]]=Result
    k=i
    i=1 #maybe unnecessary
    while (length(model[[i]])!=0){
      i=i+1
    }
    model[[Result[[5]][1]]]=Result[[1]]
    model[[Result[[5]][2]]]=Result[[2]]
    
    model=split_node(X,Y,m_feature,Index_left,model[[k]][[5]][1],model,min_leaf,Inv_Cov_Y,Command)
    
    model=split_node(X,Y,m_feature,Index_right,model[[k]][[5]][2],model,min_leaf,Inv_Cov_Y,Command)
    
    
  }else{
    ii[[1]]=matrix(Y[Index,],ncol=ncol(Y))
    model[[i]]=ii
  }
  
  
  return(model)
}
##############################################################################
single_tree_prediction <- function(Single_Model,X_test,Variable_number){
  
  
  Y_pred=matrix(  0*(1:nrow(X_test)*Variable_number)  ,nrow=nrow(X_test),  ncol=Variable_number)
  
  for (k in 1:nrow(X_test)){
    xt=X_test[k, ]
    i=1
    Result_temp=predicting(Single_Model,i,xt,Variable_number)
    Y_pred[k,]=unlist(Result_temp)
    
  }
  #Y_pred1=unlist(Y_pred, recursive = TRUE)
  #Y_pred1=matrix(Y_pred1,nrow=nrow(X_test))
  return(Y_pred)
}
#############################################################################
predicting <- function(Single_Model,i,X_test,Variable_number){
  
  Result=NULL
  
  if(length(Single_Model[[i]])==5){
    feature_no=Single_Model[[i]][[3]]
    feature_value=X_test[feature_no]
    if(feature_value<Single_Model[[i]][[4]]){  #feature value less than threshold value
      #i=i*2+1
      Result=predicting(Single_Model,Single_Model[[i]][[5]][1],X_test,Variable_number)
    }else{                                    #feature value greater than threshold value
      #i=i*2+2
      Result=predicting(Single_Model,Single_Model[[i]][[5]][2],X_test,Variable_number)
    }
  }else{
    Result=matrix(  0*Variable_number,  ncol=Variable_number)
    if (Variable_number>1){
      for (jj in 1:Variable_number){
        Result[,jj]=mean(Single_Model[[i]][[1]][,jj])
      }
    }else {
      for (jj in 1:Variable_number){
        Result[,jj]=mean(unlist(Single_Model[[i]][[1]]))
      }
    }
    
  }
  return(Result)
}