#8/10/18: fixed a bug for the enr_se column

options(echo=FALSE);
param <- commandArgs(trailingOnly=T)
annotfile  = eval(paste(text=param[1]))
resultfile = eval(paste(text=param[2]))
outfile    = eval(paste(text=param[3]))

if (length(param)!=3) {
  cat("\nError: three arguments must be supplied.\n")
  cat("ARG1 : the file containing the sum of each annotation\n")
  cat("       by quantile of the continuous annotation\n")
  cat("ARG2 : Specify the prefix of the ldsc result outputs\n")
  cat("       (.results and .part_delete files)\n")
  cat("ARG3 : Specify the output filename.\n\n")
  stop("three arguments must be supplied.\n", call.=FALSE)
}

compute_h2 <- function(baseline_chunk,tau) {
	apply(tau*t(baseline_chunk$baseline_chunk),2,sum)
}

se_jacknife <- function(theta,theta_j){
	theta_hatJ=sum(theta-theta_j)+sum(theta_j/200)
	tau=200*theta-199*theta_j
	sqrt((1/200)*sum((tau-theta_hatJ)**2/(199)))
}

#1) Read annotfile
data=read.table(annotfile,h=F)
M                = NULL
M$chunk_size     = as.numeric(data[1,-1])
M$baseline_chunk = t(data[-1,-1])
rm(data)

#2) Read results file and delete file
data=read.table(paste(resultfile,".results",sep=""),h=T)
mytau=as.numeric(data$Coefficient)
rm(data)
jacknife = read.table(paste(resultfile,".part_delete",sep=""),h=F)

#3) Compute h2g per quantile
h2g       = compute_h2(M,mytau)
h2g_prop  = h2g/sum(h2g)
#h2g_enr_z test whether the enrichment of a quintile is significantly different 
#from the union of all the others (see Finucane et al. 2015 Nat Genet
h2g_enr_z = (h2g/M$chunk_size) - ((sum(h2g)-h2g)/(sum(M$chunk_size)-M$chunk_size)) 

#4) Compute h2g for each Jacknife estimates
h2jacknife       = NULL; 
h2jacknife_prop  = NULL; 
h2jacknife_enr_z = NULL;
for (i in 1:nrow(jacknife)){
	tau_i=NULL; for (j in 1:ncol(jacknife)) {tau_i=c(tau_i,jacknife[i,j])}
	thish2g          = compute_h2(M,tau_i)
	h2jacknife       = rbind(h2jacknife       , thish2g ); 
	h2jacknife_prop  = rbind(h2jacknife_prop  , thish2g/sum(thish2g) );
	h2jacknife_enr_z = rbind(h2jacknife_enr_z , (thish2g/M$chunk_size) - ((sum(thish2g)-thish2g)/(sum(M$chunk_size)-M$chunk_size)) );
}

#5) Compute h2g std err per quantile
h2g_sd       = NULL;
h2g_prop_sd  = NULL;
h2g_enr_z_sd = NULL;
for (i in 1:length(h2g)) {
	h2g_sd       = c(h2g_sd      ,se_jacknife(h2g[i]       ,h2jacknife[,i]))
	h2g_prop_sd  = c(h2g_prop_sd ,se_jacknife(h2g_prop[i]  ,h2jacknife_prop[,i]))
	h2g_enr_z_sd = c(h2g_enr_z_sd,se_jacknife(h2g_enr_z[i] ,h2jacknife_enr_z[,i]))
}

#6) write outputs
#out = cbind(h2g,h2g_sd,h2g_prop,h2g_prop_sd,(h2g/sum(h2g))/(M$chunk_size/sum(M$chunk_size)),(h2g_sd/sum(h2g))/(M$chunk_size/sum(M$chunk_size)),2*pnorm(-abs(h2g_enr_z/h2g_enr_z_sd)))
out = cbind(h2g,h2g_sd,h2g_prop,h2g_prop_sd,h2g_prop/(M$chunk_size/sum(M$chunk_size)),h2g_prop_sd/(M$chunk_size/sum(M$chunk_size)),2*pnorm(-abs(h2g_enr_z/h2g_enr_z_sd)))
colnames(out)=c("h2g","h2g_se","prop_h2g","prop_h2g_se","enr","enr_se","enr_pval")
			
write.table(file=outfile,out,quote=F,col.names=T,row.names=F,sep="\t")
