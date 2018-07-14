# View_enhanced_ALS
The implementation of VALS

View-enhanced eALS performs well by integrating users' view data in E-commerce websites as an intermediate feedback. This is our official implementation for the paper: 

Jingtao Ding, Guanghui Yu, Xiangnan He, Yuhan Quan, Yong Li, Tat-Seng Chua, Depeng Jin and Jiajie Yu, **Improving Implicit Recommender Systems with View Data**, In Proceedings of IJCAI'18.

If you use the codes, please cite our paper . Thanks!

# Requirements
JAVA

# Currently the VALS, eALS, BPR and MC-BPR are provided. We will update the code of other baselines in a near future.

# Quick Start

## run VALS

java -jar <name_of_jar>.jar main_MF <path_of_purchase_file> vieweALS <value_s0> False True <number_factors> <number_iterations> <reg_value> 0 <path_of_view_file> <value_c0> 0 <value_gamma1> <value_gamma2> 

## run eALS

java -jar <name_of_jar>.jar main_MF <path_of_purchase_file> fastals <value_s0> False True <number_factors> <number_iterations> <reg_value> 0

## run BPR

java -jar <name_of_jar>.jar main_MF <path_of_purchase_file> bpr <learning_rate> False True <number_factors> <number_iterations> <reg_value>

## run MC-BPR

java -jar <name_of_jar>.jar mfbpr_pos 0 0 <learning_rate> False True <number_factors> <number_iterations> <reg_value> 0 <path_of_purchase_file> <path_of_view_file> false <value_beta1> <value_beta2> 
