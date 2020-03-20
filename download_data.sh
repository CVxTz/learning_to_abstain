wget -c https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget -c https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
# transform the that disgusting .mat file into a csv
python learning_to_abstain/dat_to_csv.py
