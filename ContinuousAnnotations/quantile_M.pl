#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long; 
use IO::Uncompress::Gunzip qw($GunzipError);

sub print_header
{
  print "\n";
  print "\@------------------------------------------------------------------@\n";
  print "|                  Compute sum of each annotation                  |\n";
  print "|              by quantile of continuous annotations               |\n";
  print "|                     sgazal\@hsph.harvard.edu                      |\n";
  print "|                      last modified: 08/10/18                     |\n";
  print "\@------------------------------------------------------------------@\n";
  print "\n";
}

sub print_help
{
  print "This script allows computing the sum of each annotation by quantile\n";
  print "of a continuous annotation\n\n";
  print "usage: quantile_M.pl [--help]\n"; 
  print "       [--frqfile-chr FRQFILE_CHR] [--ref-annot-chr REF_ANNOT_CHR]\n"; 
  print "       [--annot-header ANNOT_HEADER] [--nb-quantile NB_QUANTILE]\n"; 
  print "       [--maf MAF_THRESHOLD] [--out OUTFILE] [--exclude0]\n\n"; 
  print "optional arguments:\n";
  print "--help\n";
  print "       Print this help message and exit\n";
  print "--frqfile-chr FRQFILE_CHR\n";
  print "       Prefix for allele frequency files split over chromosome.\n";
  print "       These files should be in the format FRQFILE_CHRchr.frq or\n";
  print "       FRQFILE_CHRchr.frq.gz\n";
  print "--ref-annot-chr REF_ANNOT_CHR\n";
  print "       Prefix for annotation files split over chromosome. These\n";
  print "       files should be in the format REF_LD_CHRchr.annot.gz.\n";
  print "       Several annotation files can be combined when seperated\n";
  print "       by a comma. The corresponding files need to have the same\n";
  print "       SNPs in the same order.\n";
  print "--annot-header ANNOT_HEADER\n";
  print "       Specify the header of your annotation of interest in the\n";
  print "       REF_ANNOT_CHR files.\n";
  print "--nb-quantile NB_QUANTILE\n";
  print "       Specify the number of quantiles to generate (5 by default).\n";
  print "--maf MAF_THRESHOLD\n";
  print "       Specify the MAF threshold to use to include reference SNPs\n";
  print "       (0.05 by default)\n";
  print "--out OUTFILE\n";
  print "       Specify the output filename.\n";
  print "--exclude0\n";
  print "       Do not consider continuous value equals to 0 (in special case\n";
  print "       where the annotation is quantile normalized and 0 tags missing\n";
  print "       value).\n";
  exit;
}

my($frqfile_chr)=""; my($ref_annot_chr)=""; my($annot)=""; my($nb_quantile)=5; my($maf_threshold)=0.05; my($out_file)=""; my($i); my($j); my($k); my($printhelp); my($exclude0);
GetOptions(
  "help"            => \$printhelp,
  "frqfile-chr=s"   => \$frqfile_chr, 
  "ref-annot-chr=s" => \$ref_annot_chr, 
  "annot-header=s"  => \$annot,
  "nb-quantile=s"   => \$nb_quantile,
  "maf=s"           => \$maf_threshold,
  "out=s"           => \$out_file,
  "exclude0"        => \$exclude0
);

print_header();
if (defined($printhelp)){ print_help() }
if (($frqfile_chr eq "") || ($ref_annot_chr eq "") || ($annot eq "") || ($out_file eq "")) {die "\nERROR! --freqfile-chr, --ref-annot-chr, --annot-header and --out options are mandatory.\n"}

#Step 0: find annot in ref-ld
my($annot_file)=""; 
my($annot_colm)="";
my($total_nb_annotation)=0;
my(@list_ref_annot)=split(/,/, $ref_annot_chr);
my($nb_ref_annot)  =$#list_ref_annot+1;
print "ref-ld-chr   : $nb_ref_annot file(s)\n";
for ($i=0; $i<=$#list_ref_annot ; $i++) {
	print "               $list_ref_annot[$i]\n"; 
	my $IN = IO::Uncompress::Gunzip->new( " $list_ref_annot[$i]22.annot.gz" ) or die "IO::Uncompress::Gunzip failed: $GunzipError\n";
	my(@line)=split ' ', <$IN>;
	for ($j=0; $j<=$#line ; $j++) {
		if($line[$j] eq $annot){
			$annot_file=$list_ref_annot[$i];
			$annot_colm=$j+1;
		}
	}
	close $IN;
	$total_nb_annotation=$total_nb_annotation+$#line-3;
}
print "               $total_nb_annotation total annotations are present in annotation files.\n";
print "annot-header : $annot\n";
if($annot_file eq "") {die "\nERROR! Annotation $annot has not been find in any annotation files.\n"}
print "               present in column $annot_colm of annotation files $annot_file\n";
print "frqfile-chr  : $frqfile_chr\n";
print "nb-quantile  : $nb_quantile\n";
print "maf          : $maf_threshold\n";
if ($maf_threshold<0 || $maf_threshold>0.5) {die "ERROR! MAF sould be between O and 0.5\n\n"}
print "exclude0     : ";
if (defined($exclude0)) {print "yes\n"} else {print "no\n"}
print "out          : $out_file\n";
print "\n";

#Step 1: Stock MAF from $frqfile_chr files
print "Step 1: Stock MAF from $frqfile_chr files\n";
my(@MAF)=();
my($chr);
my($nbSNPs)=0;
my($nbcommonSNPs)=0;
for ($chr=1; $chr<=22 ; $chr++) {	
	if (-e "$frqfile_chr$chr.frq") {
		open(IN,"$frqfile_chr$chr.frq") or die "\nERROR! Cannot open $frqfile_chr$chr.frq.\n";
		while (<IN>) {
			chomp $_; my(@line)=split;
			if($line[4] ne "MAF"){
				if ($line[4]>=$maf_threshold && $line[4]<=(1-$maf_threshold)) { push(@MAF,1); $nbcommonSNPs++} else { push(@MAF,0) };
				$nbSNPs++;
			}
		}
		close IN;
	} elsif (-e "$frqfile_chr$chr.frq.gz")  {
		my $IN = IO::Uncompress::Gunzip->new( "$frqfile_chr$chr.frq.gz" ) or die "IO::Uncompress::Gunzip failed: $GunzipError\n";
		while (<$IN>) {
			chomp $_; my(@line)=split;
			if($line[4] ne "MAF"){
				if ($line[4]>=$maf_threshold && $line[4]<=(1-$maf_threshold)) { push(@MAF,1); $nbcommonSNPs++} else { push(@MAF,0) };
				$nbSNPs++;
			}
		}
		close $IN;
	} else {
		die "\nERROR! Cannot open $frqfile_chr$chr.frq(.gz).\n";
	}
}
print "        $nbcommonSNPs/$nbSNPs SNPs with MAF>=$maf_threshold\n\n";

#Step 2: Stock continuous annotations from $annot_file files
print "Step 2: Stock continuous annotations from $annot_file files\n";
print "        and compute quantiles\n";
my($Q);
my(@annot_value)=();
my(@annot_value_all)=();
my($cpt)=0;
$j=$annot_colm-1;
for ($chr=1; $chr<=22 ; $chr++) {
	my $IN = IO::Uncompress::Gunzip->new( " $annot_file$chr.annot.gz" ) or die "IO::Uncompress::Gunzip failed: $GunzipError\n";
	while (<$IN>) {
		chomp $_; my(@line)=split;
		if($line[$j] ne $annot){
			if ($MAF[$cpt]==1){ 
				if (!defined($exclude0) || $line[$j]!=0) {
					push(@annot_value,$line[$j])
				}
			}
			push(@annot_value_all,$line[$j]);
			$cpt++;
		}
	}
	close IN;
}
my($nb_annot_value)=$#annot_value+1;
print "        $nb_annot_value $annot values with MAF>=$maf_threshold\n\n";

@annot_value = sort {$a <=> $b} @annot_value; 

my(@Qvect)=();
for ($i=0; $i<=$nb_quantile ; $i++) {
	$Q=$annot_value[sprintf("%.0f",($i*($#annot_value)/$nb_quantile))];
	print "Q$i = $Q\n";
	push(@Qvect,$Q);
}

#substract epsilon to $Qvect[0]
$Qvect[0]=$Qvect[0]-0.0000000001;

#Step 3: Compute sum of each annotation in each quantile
print "\nStep 3: Compute sum of each annotation in each quantile\n";
#Initialize matrix
my @matrix;
my @thismatrix;
my @nb_per_Q;
my @list_annot;
#
my($cptQ);
for ($k=0; $k<=$#list_ref_annot ; $k++) {
	print "        Reading $list_ref_annot[$k]\n";
	$cpt=0;
	@thismatrix=();
	@nb_per_Q=();
	for ($chr=1; $chr<=22 ; $chr++) { 
		my($cpt0)=0;
		my $IN = IO::Uncompress::Gunzip->new( " $list_ref_annot[$k]$chr.annot.gz" ) or die "IO::Uncompress::Gunzip failed: $GunzipError\n";
		while (<$IN>) {
			chomp $_; my(@line)=split;
			if($cpt0>0){
				if ($MAF[$cpt]==1){
					if (!defined($exclude0) || $annot_value_all[$cpt]!=0) {
						$cptQ=-1;
						while ($annot_value_all[$cpt]>$Qvect[$cptQ+1]) {$cptQ++}		
						for ($i=0; $i<($#line-3) ; $i++) {
							$thismatrix[$i][$cptQ]=$thismatrix[$i][$cptQ]+$line[$i+4];
						}
						$nb_per_Q[$cptQ]=$nb_per_Q[$cptQ]+1;
					}
				}
				$cpt++;
			} 
			if($cpt==0) {
				#initialize thismatrix and nb_per_Q + update list_annot
				for ($i=0; $i<($#line-3) ; $i++) {
					for ($j=0; $j<$nb_quantile ; $j++) {				
						$thismatrix[$i][$j]=0;
					}
					push(@list_annot,$line[$i+4]);
				}
				@nb_per_Q=(0) x $nb_quantile;
			}
			$cpt0=1;
		}
		close $IN;
	}
	@matrix = (@matrix,@thismatrix);
}

open(OUT,">$out_file") || die "\nCannot create $out_file file.\n";
printf OUT "N";	
for ($j=0; $j<$nb_quantile ; $j++) {
printf OUT "	$nb_per_Q[$j]";	
}
printf OUT "\n";
for ($i=0; $i<$total_nb_annotation ; $i++) {
	printf OUT "$list_annot[$i]";
	for ($j=0; $j<$nb_quantile ; $j++) {
		printf OUT "	$matrix[$i][$j]";	
	}
	printf OUT "\n";
}
close OUT;
