import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from abyss.glob import glob
from keras.regularizers import l2
"""
Function which returns you the allele frequency of a snp
"""
def count_AF(total):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(total.columns, columns = ['snp_name'])
    total = np.where(total == 0, -0.5, total)
    minor = np.where(total == 1., 0, total)
    len_min = minor.shape[0]*-1
    df['total_af'] = minor.sum(axis = 0)/len_min
    return df

"""
Mini function to find the exact ratio of signal to noise to have a statistical power of 80%
depending on the AF of the SNP, low AF -> High effect
"""
def find_power_vals(pwr):
    df_ratios_power = pd.read_pickle(f"{glob.PATH_usefull}/df_effect_vs_power.pkl")
    concated = pd.DataFrame()
    for col in list(df_ratios_power.columns)[0:1]:
        concated['power'] = df_ratios_power[col]
        concated['maf'] = np.ones(len(df_ratios_power[col])) * float(col)
        concated['ratio'] = df_ratios_power.index
    for col in list(df_ratios_power.columns)[1:]:
        temp = pd.DataFrame()
        temp['power'] = df_ratios_power[col]
        temp['maf'] = np.ones(len(df_ratios_power[col])) * float(col)
        temp['ratio'] = df_ratios_power.index
        concated = pd.concat([concated, temp])
        rslt_df = concated[pwr-2 < concated['power']]
        rslt_df = rslt_df[rslt_df['power'] < pwr+2]
        
    mafs = np.array(rslt_df['maf'])
    ratios = np.array(rslt_df['ratio'])
    a = (ratios.min()-ratios.max())/(mafs.max())**2
    b = -2*a*mafs.max()
    c = ratios.min()
    return [a, b, c]


"""
Mini_function to make sure the labels aren't unreadable
will transform an allele frequency looking like: 0.000758962 into 0.00075, this will also reduce the number of labels
"""
def fa(af):
    def num_of_zeros(n):
        s = '{:.16f}'.format(n).split('.')[1]
        return len(s) - len(s.lstrip('0'))

    minimum = af.min()
    multi = 10**num_of_zeros(minimum)
    return list(round(round(af*multi,2)/multi,4))

"""
Function that gives you a compararaison between one snp with a certain population stratification scenario
df_geno (one column pandas dataframe containing the genotype of one snp)
minimum_power is the minimum power you want
iteration is how many times the analysis needs to run
df_corr is a pandas df with the correction covs
path_geno is the path to the genotype f"data/genotype/segmented_outliers/" (so pre chroms)
"""
def popstratcorrector(df_geno, df_pheno_pop, df_covs):
    return ''
    
def dupplicate_collumns(df1, df2):
    return list(set(list(df1.columns)) & set(list(df2.columns)))
"""
Having layers as multiples of 8 helps considerably with the computing power of the model
"""

def bitify(num):
    return round(num/8)*8

"""
Create little bloks of neural nets
"""
initializer = 'he_normal'

def create_blok(x, units, activation = 'elu', kernel_initializer=initializer):
    reg_lambda=0.01
    x = Dense(units, kernel_regularizer=l2(reg_lambda))(x)
    x = Activation(activation)(x)
    return x

def metrics_viewer(model):
    acu = model.history['accuracy']
    val_acu = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochsticks = range(len(acu))
    plt.figure(figsize = (15, 10))
    ax1 = plt.subplot(311)
    ax1.plot(epochsticks, acu, 'b', label = 'Training Accuracy')
    ax1.plot(epochsticks, val_acu, 'g', label = 'Validation Accuracy')
    ax1.set_title('Training and validation Accuracy')
    ax1.grid()
    ax1.legend()
    ax2 = plt.subplot(312, sharex = ax1)
    ax2.plot(epochsticks, loss, 'b', label = 'Training loss')
    ax2.plot(epochsticks, val_loss, 'g', label = 'Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.grid()
    ax2.legend()
    return plt

def population_viewer(humans, hidden_repr, title, path_output):
    
    humans['ancestry'] = humans['super_pop']
    humans['ancestry'].mask(humans['super_pop'] == 'AFR', 0, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'AMR', 1, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'EAS', 2, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'EUR', 3, inplace=True)
    humans['ancestry'].mask(humans['super_pop'] == 'SAS', 4, inplace=True)
    for i in range(np.shape(hidden_repr)[1]):
        humans[f"dim{i+1}"] = hidden_repr[:,i]

    palette = 'mako'
    plt.figure(figsize = (15, 10))
    sns.scatterplot(
        data=humans,
        x="dim1",
        y="dim2",
        hue="super_pop",
        palette = palette).set(title=title)
    plt.savefig(f"{path_output}/super_population.png")
    plt.close()
    afr = humans.loc[humans['ancestry'] == 0]
    amr = humans.loc[humans['ancestry'] == 1]
    eas = humans.loc[humans['ancestry'] == 2]
    eur = humans.loc[humans['ancestry'] == 3]
    sas = humans.loc[humans['ancestry'] == 4]

    import warnings
    warnings.filterwarnings('ignore')

    palette_afr = sns.blend_palette(['limegreen', 'black'], len(list(afr['pop'].unique()))) # afr
    palette_amr = sns.blend_palette(['salmon','darkred'], len(list(amr['pop'].unique()))) # amr
    palette_eur = sns.blend_palette(['yellow','darkorange'], len(list(eur['pop'].unique()))) #europe
    palette_eas = sns.blend_palette(['turquoise', 'midnightblue'], len(list(eas['pop'].unique()))) # eas
    palette_sas = sns.blend_palette(['lightpink','darkviolet'], len(list(sas['pop'].unique()))) # sas

    fig, ax = plt.subplots(figsize=(12, 6))
    afr_plot = sns.scatterplot(
        data=afr,
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_afr)
    h1, l1 = afr_plot.get_legend_handles_labels()
    legend1 = ax.legend(h1, l1,
                    loc="lower left", title="Africans")
    amr_plot = sns.scatterplot(
        data=amr,
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_amr)
    h2, l2 = amr_plot.get_legend_handles_labels()
    l2 = [item for item in l2 if item not in l1]
    h2 = [item for item in h2 if item not in h1]

    legend2 = ax.legend(h2, l2,
                        loc="upper left", title="Americans")
    eur_plot = sns.scatterplot(
        data=eur,
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_eur)
    h3, l3 = eur_plot.get_legend_handles_labels()
    l3 = [item for item in l3 if item not in l1]
    l3 = [item for item in l3 if item not in l2]
    h3 = [item for item in h3 if item not in h1]
    h3 = [item for item in h3 if item not in h2]
    legend3 = ax.legend(h3, l3,
                        loc="upper right", title="Europeans")
    eas_plot = sns.scatterplot(
        data=eas,
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_eas)
    h4, l4 = eas_plot.get_legend_handles_labels()
    l4 = [item for item in l4 if item not in l1]
    l4 = [item for item in l4 if item not in l2]
    l4 = [item for item in l4 if item not in l3]

    h4 = [item for item in h4 if item not in h1]
    h4 = [item for item in h4 if item not in h2]
    h4 = [item for item in h4 if item not in h3]
    legend4 = ax.legend(h4, l4,
                        loc="lower right", title="East-asians")
    sas_plot = sns.scatterplot(
        data=sas,
        x="dim1",
        y="dim2",
        hue="pop",
        palette = palette_sas)

    h5, l5 = sas_plot.get_legend_handles_labels()
    l5 = [item for item in l5 if item not in l1]
    l5 = [item for item in l5 if item not in l2]
    l5 = [item for item in l5 if item not in l3]
    l5 = [item for item in l5 if item not in l4]

    h5 = [item for item in h5 if item not in h1]
    h5 = [item for item in h5 if item not in h2]
    h5 = [item for item in h5 if item not in h3]
    h5 = [item for item in h5 if item not in h4]
    legend5 = ax.legend(h5, l5,
                        loc="upper center", title="South-asians")

    plt.tight_layout()
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)
    ax.add_artist(legend4)
    ax.add_artist(legend5)
    plt.savefig(f"{path_output}/sub_population.png")
    plt.close()
    
    plt.figure(figsize = (15, 10))
    sns.scatterplot(
    data=afr,
    x="dim1",
    y="dim2",
    hue="pop",
    palette = palette_afr).set(title=f"{title} of African Populations")
    plt.savefig(f"{path_output}/afr_population.png")
    plt.close()
    
    plt.figure(figsize = (15, 10))
    sns.scatterplot(
    data=amr,
    x="dim1",
    y="dim2",
    hue="pop",
    palette = palette_amr).set(title=f"{title} of American Populations")
    plt.savefig(f"{path_output}/amr_population.png")
    plt.close()
    
    plt.figure(figsize = (15, 10))
    sns.scatterplot(
    data=eur,
    x="dim1",
    y="dim2",
    hue="pop",
    palette = palette_eur).set(title=f"{title} of European Population")
    plt.savefig(f"{path_output}/eur_population.png")
    plt.close()
    
    plt.figure(figsize = (15, 10))
    sns.scatterplot(
    data=sas,
    x="dim1",
    y="dim2",
    hue="pop",
    palette = palette_sas).set(title=f"{title} of South Asian Population")
    plt.savefig(f"{path_output}/sas_population.png")
    plt.close()
    
    plt.figure(figsize = (15, 10))
    sns.scatterplot(
    data=eas,
    x="dim1",
    y="dim2",
    hue="pop",
    palette = palette_eas).set(title=f"{title} of East Asian Population")
    plt.savefig(f"{path_output}/eas_population.png")
    plt.close()
    
def create_plot(hue, data, x, y, title, plot_path):
    from collections import Counter

    import matplotlib.pyplot as plt
    import seaborn as sns

    from abyss.glob import glob

    pal_len = len(Counter(hue).keys())
    palette = sns.blend_palette(glob.SEABORN_COLORS[100:-2], pal_len)
    plt.figure(figsize = (15, 10))
    sns.scatterplot(
    data=data,
    x=x,
    y=y,
    hue = hue,
    palette = palette, legend=True).set(title=title)
    plt.savefig(plot_path)

def y_pred_model(dims, y):
    # divide in train - test sets
    import tensorflow as tf

    X = tf.convert_to_tensor(dims, dtype=tf.float32)
    len_test = round(dims.shape[0]/10)*3
    len_train = dims.shape[0] - len_test
    X_train = dims.head(len_train)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)

    y_train = y.head(len_train)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    X_test = dims.tail(len_test)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    y_test = y.tail(len_test)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='elu'),
    tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse','accuracy'])

    history = model.fit(X_train,
          y_train,
          validation_data=(X_test, y_test),
          epochs = 250,
          verbose = 0)

    metric_plt = metrics_viewer(history)
    metric_plt.savefig(f"temporary_plots/metrics_y_pred.png")
    metric_plt.close()

    return model(X).numpy()
