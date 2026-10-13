"""
Abyss comes with plink pre-installed so those commands should be usable from the get-go
If (like us) you have files with samples that are separated in ancestries from the 1000 Genomes project you first have to merge them altogether.
For that create a merge_list.txt file in the folder containing your separated samples with all the files that you need to merge like so:
"""


# IMPORT LIBRARIES
import time
import os
from abyss.glob import glob

# Define paths
print(glob.PATH_GENO)
path_input = glob.PATH_GENO + "/01_raw"
path_output = glob.PATH_GENO + "/02_per_chrom"
print(path_output)

# Check if you have multiple files that need merging
# If you do, put their names into the merge_list
os.system("unzip plink_linux_x86_64_20230116.zip")
time.sleep(1)
os.system("A")
os.system(f"touch {path_input}/merge_list.txt")
os.system(f"mkdir {path_output}")

# Create a merged total population file for each chromosome called total_pop_chr_#chrom
path_plink = os.getcwd()
files = [f.split('.fam')[0] for f in os.listdir(path_input) if f.endswith(".fam")]
print(files)
for i in range(22):
    os.system("pwd")
    print(i+1)
    os.system(f"cd {path_input}; {path_plink}/plink --merge-list {path_input}/merge_list.txt --make-bed --chr {i+1} --out {path_output}/total_pop_chr_{i+1}")
