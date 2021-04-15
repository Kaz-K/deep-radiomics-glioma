import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import bartlett
import matplotlib.pyplot as plt


radiomics_path = './results/radiomics/radiomics.csv'
name_mapping_path = './data/name_mapping.csv'


if __name__ == '__main__':
    mapping = pd.read_csv(name_mapping_path, header=0)[['Grade', 'BraTS_2019_subject_ID']]
    radiomics = pd.read_csv(radiomics_path, header=0, index_col=0)

    df = radiomics.merge(mapping, left_index=True, right_on='BraTS_2019_subject_ID')

    print(df.groupby('Grade').mean())
    print(df.groupby('Grade').var())

    lgg = df[df['Grade'] == 'LGG']
    hgg = df[df['Grade'] == 'HGG']

    print(bartlett(lgg['ALL_VoxelVolume'], hgg['ALL_VoxelVolume']))
    print(ttest_ind(lgg['ALL_VoxelVolume'], hgg['ALL_VoxelVolume'], equal_var=False))

    df.groupby('Grade')['ALL_VoxelVolume'].apply(
        lambda x: sns.distplot(x, bins=50, hist=True, rug=False, label=x.name)
    )

    plt.xlabel('Voxel volume')
    plt.ylabel('KDE')
    plt.show()
