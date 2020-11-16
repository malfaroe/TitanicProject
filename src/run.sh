#Script for running multiple folders with an algorithm
clear
echo "starting shell script"
python train.py --fold 0 --model decision_tree_gini
echo "ending shell script"

