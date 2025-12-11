import polars as pl
import pandas as pd
import os
import urllib
from tqdm import tqdm
import gzip
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Progress bar

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import numpy as np
from collections import Counter


from functools import partial

import polars as pl
from tqdm import tqdm
import pandas as pd
import numpy as np
from glob import glob
import sys

dna_code = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGC": "C",
    "TGT": "C",
    "TGA": "*",
    "TGG": "W",
}
AA = set(dna_code.values())
idxes = [13,49,0,4,52,7,63,29,19,35,43,14,3,33,57,24,60,5,56,59,21,11,28,55,62,58,30,26,22,
         48,17,12,10,23,44,15,20,27,40,9,8,54,38,37,16,36,32,18,34,39,31,46,2,42,53,6,61,25,47,50,51,45,41,1]
dna_codon_idx = {k:i for i,k in zip(idxes, dna_code.keys())}
dna_codon_idx['---'] = -1

# Define the genetic code as a dictionary mapping codons to amino acids
rna_code = {
    "UUU": "F",
    "UUC": "F",  # Phenylalanine
    "UUA": "L",
    "UUG": "L",  # Leucine
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",  # Serine
    "UAU": "Y",
    "UAC": "Y",  # Tyrosine
    "UAA": "*",
    "UAG": "*",  # Stop codons
    "UGA": "*",  # Stop codon
    "UGU": "C",
    "UGC": "C",  # Cysteine
    "UGG": "W",  # Tryptophan
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAU": "H",
    "CAC": "H",  # Histidine
    "CAA": "Q",
    "CAG": "Q",  # Glutamine
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",  # Methionine (start codon)
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

rna_codon_idx = {k:i for i,k in zip(idxes, rna_code.keys())}

def translate(seq, codon_dict):
    """
    Translate an RNA sequence into a protein sequence.
    Stops translation when a stop codon ('*') is encountered.
    """
    is_str = isinstance(list(codon_dict.values())[0], str)
    if is_str:
        protein = ""
    else:
        protein = []
    # Process the RNA sequence three nucleotides (codon) at a time.
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        # Look up the codon in the genetic code dictionary.
        
        if is_str:
            amino_acid = codon_dict.get(codon, "?")
            protein += amino_acid
        else:
            amino_acid = codon_dict[codon]
            protein.append(amino_acid)
            
    return protein


# The file can be downloaded from https://ftp.ncbi.nlm.nih.gov/genomes/refseq/assembly_summary_refseq.txt
# !wget https://ftp.ncbi.nlm.nih.gov/genomes/refseq/assembly_summary_refseq.txt

#NOTE: that this file may be changing. The models were trained with release version 228 downloaded on Mar-10-2025
meta = pd.read_table('assembly_summary_refseq.txt', skiprows=1,low_memory=False)
meta = meta.loc[(meta['refseq_category']=='reference genome')]

meta.groupby('group')['#assembly_accession'].apply(lambda x:x.unique().shape[0])

import json
taxids_to_remove = sum([v for _,v in json.load(open('taxids_to_remove_bac.json')).items()], [])

meta = meta.loc[~meta['taxid'].isin(taxids_to_remove)]
meta.groupby('group')['#assembly_accession'].apply(lambda x:x.unique().shape[0])


def download_files(ftp_paths, destination_folder='ncbi_refseq_reference/raw'):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for ftp_path in tqdm(ftp_paths):
        
        parent_dir_name = ftp_path.split('/')[-1]
        curr_dir = os.path.join(destination_folder, parent_dir_name[:10])
        if not os.path.exists( curr_dir):
            os.makedirs(curr_dir)
        file_name = parent_dir_name + '_cds_from_genomic.fna.gz'
        file_url = f"{ftp_path}/{file_name}"
        local_file_path = os.path.join(curr_dir, file_name)
        
        try:
            urllib.request.urlretrieve(file_url, local_file_path)
        except Exception as e:
            print(f"Failed to download {file_url}. Error: {e}")

# download_files(meta['ftp_path'].tolist())




# Load tax-id to name mapping
tax_id_to_name = {}
with open('names.dmp', 'r') as f:
    for line in f:
        fields = line.strip().split('\t|\t')
        tax_id = int(fields[0])
        name = fields[1]
        if name != 'scientific name':
            tax_id_to_name[tax_id] = name

# Load tax-id to parent mapping
tax_id_to_parent = {}
with open('nodes.dmp', 'r') as f:
    for line in f:
        fields = line.strip().split('\t|\t')
        tax_id = int(fields[0])
        parent_id = int(fields[1])
        tax_id_to_parent[tax_id] = parent_id

import pandas as pd

# Load names.dmp into a dictionary
names_dict = {}
with open('names.dmp', 'r') as f:
    for line in f:
        fields = line.strip().split('\t|\t')
        tax_id = int(fields[0])
        name = fields[1]
        if name != 'scientific name':
            names_dict[name] = tax_id

# Map organisms to taxonomy IDs
organisms = ["Xenopus tropicalis", "Homo sapiens"]
tax_ids = {org: names_dict.get(org) for org in organisms}

print(tax_ids)

to_class = [names_dict['viruses'], names_dict['bacteria'],
           names_dict['Primates'], names_dict['Rodentia'], names_dict['Mammalia'], 
            names_dict['Phage sp.'],
           names_dict['invertebrate metagenome'], names_dict['Fungi'],
           names_dict['Viridiplantae'], names_dict['vertebrates']]

to_class_names = ['viruses','bacteria','Primates','Rodents','Mammals','Phage',
                  'invertebrate', 'fungi','plants', 'vertebrates', 'others']

ref_classes = {}
for t in tqdm(meta['taxid'].values):
    tt = t
    if t not in tax_id_to_parent:
        continue
    while t not in to_class:
        nt = tax_id_to_parent[t]
        if nt == t:
            break
        else:
            t = nt
    if t in to_class:
        ref_classes[tt]= to_class_names[to_class.index(t)]
    else:
        ref_classes[tt]= to_class_names[-1]
        
        


def check_sequence(seq):
    if len(seq) %3 == 0:
        aa = translate(seq, dna_code)
        if len(set(aa).difference(AA)) == 0:
            if '*' not in aa.rstrip('*'):
                return True
    return False


assembly_to_taxid = {a:b for a,b in meta[['#assembly_accession','taxid']].values}
assembly_to_group = {a:b for a,b in meta[['#assembly_accession','group']].values}
def process_file(fn, output_dir):
    # Extract the assembly ID
    match = re.search(r'GCF_\d+\.\d+', fn)
    if not match:
        print(f"No match found for assembly in filename: {fn}")
        return
    assembly = match.group(0)
    taxid = assembly_to_taxid[assembly]
    # Create output file for this assembly
    group_name = assembly_to_group[assembly] if ref_classes[taxid] != 'Primates' else ref_classes[taxid]
    output_file = os.path.join(output_dir, f"{group_name}_{assembly}.csv")

    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            return taxid, group_name, df.shape[0]
        except Exception as e:
            print(f"Exception in file {output_file}, reprocess.")

    
    with gzip.open(fn, 'rt') as f: #, open(output_file, 'w') as outfile:
        # Write the CSV header
        parsed_data = []
        # Process entries and write directly to the file
        for entry in f.read().split('\n>'):
            lines = entry.strip().split("\n")
            header = lines[0]
            sequence = "".join(lines[1:])  # Combine all lines of the sequence
            if check_sequence(sequence):
                # Extract the ID (everything after 'lcl|' up to the first space)
                match = re.search(r'lcl\|([^ ]+)', header)
                if match:
                    id_ = match.group(1)
                    # Write the entry directly to the output file
                    parsed_data.append([assembly,group_name, taxid,id_,sequence])
                else:
                    print(f"Invalid header in file {fn}: {header}")
                    break
    parsed_data = pd.DataFrame(parsed_data)
    parsed_data.columns = ['assembly','group','taxid','seq_id','cds']
    parsed_data = parsed_data.drop_duplicates(subset='cds')
    parsed_data.to_csv(output_file, index=False)
    return taxid, group_name, parsed_data.shape[0]

def process_files_in_parallel(files, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    with ProcessPoolExecutor() as executor:
        # Use tqdm to track progress
        outputs = list(tqdm(executor.map(process_file, files, [output_dir] * len(files)), 
                  total=len(files), desc="Processing Files"))

    return outputs


files = sorted(glob('ncbi_refseq_reference/raw/*/*.fna.gz'))
ouputs = process_files_in_parallel(files, 'ncbi_refseq_reference/processed/')
outputs_df = pd.DataFrame(ouputs)
outputs_df.columns = ['taxid','class','count']
counts = outputs_df.groupby('class')['count'].sum().reset_index()

unique_counts = []
for group in counts['class'].values:
    
    seqs = []
    for fn in tqdm(glob(f'ncbi_refseq_reference/processed/{group}*.csv')):
        seqs.append(pl.read_csv(fn))
    seqs = pl.concat(seqs)
    seqs = seqs.sort('taxid')
    out = []
    for tid in seqs['taxid'].unique():
        temp = seqs.filter(pl.col('taxid')==tid).unique('cds')
        out.append(temp)
    out = pl.concat(out)
    # if out.shape[0] != seqs.shape[0]:
    out_fn = f'ncbi_refseq_reference/processed_grouped/{group}.csv'
    out.write_csv(out_fn)
    unique_counts.append([group, seqs.shape[0]])
counts_df = pd.DataFrame(unique_counts)
counts_df.columns = ['class','count']




labels = [f"{cls} ({cnt:,})" for cls, cnt in zip(counts_df['class'], counts_df['count'])]

# Create the pie chart
plt.pie(counts_df['count'], labels=labels, autopct='%1.1f%%')
plt.title("Distribution of sequences across organisms")
plt.savefig('ncbi_refseq_reference/sequence_distribution.png',dpi=300)
# plt.show()


meta['new_group'] = [ref_classes[t] if ref_classes[t]=='Primates' else v  
                     for a,t,v in meta[['#assembly_accession','taxid','group']].values]

org_counts = meta['new_group'].value_counts().reset_index().values
labels = [f"{label} ({count})" for label, count in org_counts if label != 'bacteria'] # Add counts to labels
sizes = list([x[1] for x in org_counts[1:] if x[0] != 'bacteria'])  # Counts as sizes for the pie chart

# Create the pie chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of organisms")
plt.axis('equal')  
plt.savefig('ncbi_refseq_reference/organism_distribution.png', dpi=300)

org_counts = meta['new_group'].value_counts().reset_index().values
labels = [f"{label} ({count})" for label, count in org_counts[:]] # Add counts to labels
sizes = list([x[1] for x in org_counts[:]])  # Counts as sizes for the pie chart

# Create the pie chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of organisms")
plt.axis('equal')  
plt.savefig('ncbi_refseq_reference/organism_distribution_with_bac.png', dpi=300)