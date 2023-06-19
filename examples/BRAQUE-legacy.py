"""BRAQUE - Bayesian Reduction for Amplified Quantisation in Umap Embedding"""

import os
import sys
import json
import random
import collections

import umap
import hdbscan
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# if you have an Intel processor, we strongly suggest you to install sklearnex
# and uncomment the following two lines
# from sklearnex import patch_sklearn
# patch_sklearn()


# SETUP ------------------------------------------------------------------------

# fix some global parameters
np.random.seed(42)
random.seed(42)
old_fontsize = plt.rcParams['font.size']
new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
plt.rcParams.update({'font.size': 15})

# select where to redirect standard output, otherwise just comment these two lines
output_file = "output.txt"
sys.stdout = open(output_file,"w", 1)

# select if you wish a verbose print or not
verbose_output = True
verboseprint = print if verbose_output else lambda *a, **k: None


# dataset
area = "PATH/TO/DATA"

# reference file
reference_file = "reference.csv"  # path to reference .csv file
correspondence_column = 'Ab'  # header of the reference file column which contains features names, may contain variants
naming_column = 'Name'        # header of the reference file column which contains features names that shall be used in plots/results
interpretative_column = 'Significance'  # header of the reference file column which contains features associated propery
importance_column = 'Lineage Defining'  # header of the reference file column which contains 1 for important features that should be used for summary plot

markers_subgroup = ''    # header of the reference file column which might opionally be used to keep only a subset of features. if used shall be a 0/1 coded column, with 1 for keeping the feature at that specific row, or 0 to exlude it. use an empty string ('') to avoid such subselection


# IMPORTANT PARAMETERS               Suggested val
perform_features_selection = True    # True

# BayesianGaussianMixture
perform_lognormal_shrinkage = True   # wether or not perform LNS (Lognormal Shrinkage preprocessing from Dall'Olio et al. https://doi.org/10.3390/e25020354
max_n_gaussians = 15                 # 15 for cores and 20 for limphnode
contraction_factor = 5.              # between 2 and 10, suggested 5, is affected by the chosen base for the logarithm

# UMAP
nn = 50                              # 50, but should not brutally affect varying in hte range 30~100
metric = 'euclidean'                 # euclidean if no 

# load premade steps if needed
load_embed = False
load_db = False
load_clusters = False
large_db_procedure = False
if large_db_procedure:
    load_embed = True
    load_db = True
    load_clusters = False


# saving directories for resulting plots
base_folder = "./"
folder = base_folder+"results/"+area+markers_subgroup+"/"
embedding_storage = base_folder+"embeddings/"+area+markers_subgroup


# start of the analysis
verboseprint(datetime.now())






### FUNCTIONS DEFINITIONS




# UTILS ---------------------------------------------------------------------------------------------


def custom_colormap_for_many_clusters(n_clusters=None, random_seed=42, 
                                      bright_threshold=0.2):
    
    """New colormap to deal properly with 20+ clusters scenarios.
    
    \nn_clusters (integer): number of clusters, each of which will correspond to a color in the resulting output
    \nrandom_seed (integer): random seed for color order, different seeds will give different color orders
    \nbright_threshold (float, between 0 an 1): value used to discard shades of white and very bright colors, the higher the less colors will be used for the colormap
    
    """
    
    import matplotlib.colors as mcolors
    
    # =============================================================================
    # New colormap to deal properly even 20+ clusters scenarios
    # =============================================================================
    
    colors_rgb_list = [(0.0, 0.0, 0.0)]  # start with black for noise
    colors_list = mcolors.CSS4_COLORS
    output_colors = list(colors_list.keys())  # use all colors
    if n_clusters is None:
        n_clusters = len(output_colors)  # 17, 120

    # shuffle colors to avoid similar colors always adjacent on list
    random.seed(random_seed)
    random.shuffle(output_colors)
    
    # remove bright colors (e.g. shades of white) and black (since we put it by hand in 1st position for noise
    for i in output_colors:
        color_rgb = 1-np.asarray(mcolors.hex2color(colors_list[i]))
        if np.sum(color_rgb**2)**0.5 < bright_threshold or color_rgb.min()==1.:  # removing also black since we put it by hand in 1st position for noise
            output_colors.remove(i)
            verboseprint("removed color", i)

    # keep only n_clusters different colors in the final colormap
    for i in output_colors[:n_clusters]:
        colors_rgb_list.append(mcolors.hex2color(colors_list[i]))

    return mcolors.ListedColormap(colors_rgb_list)


# ----------------------------------------------------------------------


def find_names(markers_names):
    
    """Substitutes columns names with pre-defined standard names contained in reference file.
    
    \nmarkers_names (string or list/array-like): either signle string to convert or list of strings to convert to standard name.
    It is important for these values to exactly correspond to a value of the reference file corresponding column
    
    NOTE: we assume that the global variables: reference, correpsondence_column, and naming_column are properly defined
    
    """
    
    official_markers_names = []
    
    # <specific for the paper>
    if area == "DatasetCodex":
        if isinstance(markers_names, str):
            return [markers_names]
        else:
            return markers_names
    
    elif isinstance(markers_names, str):  
        # only 1 iteration to perform, otherwise we cicle over the single characters within the string
        try:
            ab = [a for a in reference.loc[:, correspondence_column] if markers_names in a][0]
            official_markers_names.append(reference[reference.loc[:, correspondence_column] == ab].loc[:, naming_column].iloc[0]) 
        except IndexError:
            official_markers_names.append(markers_names)
        return np.asarray(official_markers_names)
    
    else:
        for mn in markers_names:
            try:
                ab = [a for a in reference.loc[:, correspondence_column] if mn in a][0]
                official_markers_names.append(reference[reference.loc[:, correspondence_column] == ab].loc[:, naming_column].iloc[0]) 
            except IndexError:
                official_markers_names.append(mn)
        return np.asarray(official_markers_names)

    
# -------------------------------------------------------------------------


def add_main_markers_significance_columns(db, res, find_n=3, undef_thr=0., thrs_p=0.05):

    """Add inplace to input pandas dataframe the column 'MainMarkers' and a column with their interpretation.
    
    \ndb (pandas.DataFrame): processed dataframe containing cells x markers values
    \nres (pandas DataFrame with shape n_clusters x n_features): DataFrame containing effect_sizes resulting from robust d computation by Vendekar et al. of how much each feature is overexpressed for each cluster, this object is the first output of the function 'markers_importance_per_cluster'
    \nfind_n (positive integer): how many main markers to find, at most
    \nundef_thr (non-negative float): threshold below which an effect size is never considered relevant
    \nthrs_p (float in range ]0, 1[): which significance threshold should be adopted for p_values.

    """

    prevalent_population = []
    markers = []
    db[interpretative_column] =  ""
    db['MainMarkers'] = ""
    unique_clusters = np.unique(db.loc[:, 'cluster'])

    # using threshold suggested in robust Cohen's d by Vandekar et Al (2020)
    # undef_thr = 0.1

    for _ in range(1, find_n+1):
        # find markers names for 'find_n' markers with maximum effect size
        maxmarkers = [mrkr.split(sep='_')[-1] for mrkr in res.iloc[:, :].apply(lambda row: row.nlargest(_).index[-1], axis=1)]
        verboseprint("\n\n------------------------------------------\n",
                     _, "most expressed marker per cluster:\n")
        
        for i in unique_clusters:
            # if marker has larger effect size than 'udenf_thr' it will appear in MinMarkers column, 
            # otherwise it will be considered not expressed enough and therefore ignored
            if np.max(res.loc[i]) > undef_thr:
                # add the effect size value in the label
                level = res.loc[i, res.iloc[:, :].apply(lambda row: row.nlargest(_).index[-1], axis=1)[i]]
                if level < undef_thr:
                    continue  # check, this could give error if no significant marker is found
                              # and maxmarker/prevalent population result in being shorter

                maxmarker = maxmarkers[i-min(unique_clusters)]

                
                try:
                    ab = [a for a in reference.loc[:, correspondence_column] if maxmarker in a][0]
                    pop = reference[reference.loc[:, correspondence_column] == ab].loc[:, interpretative_column].iloc[0]
                    markers.append(find_names(maxmarker)[0])  # using Name instead of Ab
                    
                    # label markers with no clear correspondence as unclear
                    if pd.isna(pop):
                        prevalent_population.append("unclear")
                        verboseprint("cluster", i, "unclear nan", maxmarker)
                    else:
                        prevalent_population.append(pop)
                        verboseprint("cluster", i, maxmarker, pop)
                except IndexError:
                    prevalent_population.append("unclear")
                    verboseprint("cluster", i, "unclear", maxmarker)

                # for larger values of 'find_n' insert some newlines, in this case every 6 markers names
                if find_n > 6:
                    markers[-1] += ':'+str(level)[:3]
                    if _%7 == 6:
                        markers[-1] += '\n'
            else:
                prevalent_population.append("unclear")
                verboseprint("cluster", i, "undefined")

            # write MainMarkers column and interpretation column
            if markers != []:
                mask = db[db.loc[:, "cluster"]==i].index.values
                db.loc[mask, interpretative_column] += " | "+prevalent_population[-1]
                db.loc[mask, "MainMarkers"] += " | "+markers[-1]
            else:
                mask = db[db.loc[:, "cluster"]==i].index.values
                db.loc[mask, interpretative_column] += " | "+prevalent_population[-1]
                db.loc[mask, "MainMarkers"] += " | "


    # nullify noise labels
    mask = db.loc[:, 'cluster'] == -1  # do not plot noise (a.k.a. removed cells)
    db.loc[:, "MainMarkers"][mask] = ''
    db.loc[:, "Significance"][mask] = ''

    verboseprint("\n\n")
    for i in np.unique(db.loc[:, "cluster"]):
        verboseprint(i, db.loc[:, "MainMarkers"][db.loc[:, "cluster"]==i].iloc[0])

    return db




# PIPELINE -------------------------------------------------------------------------------------


def features_selection(db, reference, drop_unclear=True, 
                       drop_missing=True, to_drop = ['IGNORE'], 
                       special_keeps=[]):
    
    """Perform features selection over a dataframe, given a reference file on which column to keep/discard.        
    
    \ndb (pandas.DataFrame): dataframe containing cells x markers values, on which the features selection is going to be performed
    \nreference (pandas.DataFrame): dataframe which maps db columns to different properties for each column, mandatory properties are Ab and Significance
    \ndrop_unclear (boolean): whether to drop markers with no corrispondence in the reference file
    \ndrop_missing (boolean): whether to drop markers with missing significance
    \nto_drop (list or array-like): drop markers whose column named 'significance' value, in the reference file, is part of this list
    \nspecial_keeps (list or array-like): markers whose name is in this lisy will be kept anyway if they have at least a significance and it's != IGNORE

    """

    for column in db.columns.values:
        marker = column.split(sep='_')[-1]
        try:
            # <SPECIFIC FOR THE PAPER>
            if marker == 'lambda':
                antibody = [ab for ab in reference.loc[:, correspondence_column] if marker in ab][1]  # since kappasinelambda comes at 0
            else:
                antibody = [ab for ab in reference.loc[:, correspondence_column] if marker in ab][0]

            if markers_subgroup:
                # discard markers not in the specific subgroup
                to_discard = reference[reference.loc[:, correspondence_column] == antibody].loc[: , markers_subgroup].iloc[0] == 0
                if to_discard:
                    db.drop(columns=column, inplace=True)
                    verboseprint("Dropping", column, "since this marker is not in", markers_subgroup, 'subgroup.')
                    continue

            # drop unwanted populations, and/or missing/unclear ones
            population = reference[reference.loc[:, correspondence_column] == antibody].loc[:, interpretative_column].iloc[0]

            # missing
            if drop_missing and pd.isna(population):
                db.drop(columns=column, inplace=True)
                verboseprint("Dropping", column, "due to 'no specific significance'.")

            # unwanted
            else:
                for drop_candidate in to_drop:
                    if drop_candidate in population:
                        db.drop(columns=column, inplace=True)
                        verboseprint("Dropping", column, "due to unwanted significance:", drop_candidate)

        # unclear
        except IndexError:
            if drop_unclear:
                db.drop(columns=column, inplace=True)
                verboseprint("Dropping", column, "due to 'unknown significance'.")
                

        n_markers = len(db.columns)

    return db


# ----------------------------------------------------------------------------------------------------


def lognormal_shrinkage(db, subsampling=1, max_n_gaussians=20, 
                        contraction_factor=5., populations_plot=False):
    
    """Perform Lognormal Shrinkage preprocessing over a pandas datafame.
    
    \ndb (pandas.DataFrame): dataframe containing cells x markers values, on which the lognormal shrinkage preprocessing is going to be performed
    \nsubsampling (positive integer, between 1 and len(db)): subsampling parameter, take 1 cell every N. in order to speed up gaussian mixture fitting procedure
    \nmax_n_gaussians (positive integer, >=2): maximum number of fittable lognormal distributions for a single marker, keep in mind that the higher the slower and more precise the algorithm. To tune follow guidelines from Dall'Olio et al.
    \ncontraction_factor (positive float, >1.): each gaussian in the log2 space is contracted by this factor to better separate candidate subpopulations. To tune follow guidelines from Dall'Olio et al.
    \npopulations_plot (boolean): whether or not to plot the final summary about number of candidates subpopulations for each marker, useful to tune max_n_gaussians.
    
    """
    
    from sklearn.mixture import BayesianGaussianMixture
    
    verboseprint("performing Lognormal Shrinkage transformation")
    
    # dropping constant feature, to avoid erros and because they are useless
    for column in db.columns[np.where(db.var() == 0)[0]]:
        db.drop(columns=column, inplace=True)
        n_markers -= 1
        verboseprint("Dropping", column, "due to Variance = 0. \n(It is not possible to fit Bayesian Gaussian Mixture on it, the signal is constant!)")

    # robust scaling, needed to use the same mixture parameters for all the markers, which could otherwise strongly differ in range, eìoutlier percentage, etc.
    db = db/(np.abs(db - db.mean()).mean())

    # add small constant to the db, in order to avoid log(0) related errors
    epsilon = db[db>0.].min().min()
    logdb = np.log2(db+epsilon)

    # optionally print markers to check their order/number/correctness
    verboseprint("list of genes to process")
    for i, column in enumerate(db.columns):
        verboseprint(i+1, column)
        
    with open(folder+"/preprocessed_markers.txt", 'w') as f:
        print("list of features which completed the lognormal shrinkage preprocessing step", file=f)

    n_populations = {}  # to store number of gaussians per marker information
    for i, column in enumerate(db.columns):
        verboseprint(i+1, column)

        use = logdb.loc[:, column]
        # Fitting a Bayesian Gaussian Mixture, with maximum number of gaussians = max_n_gaussian, over each marker separately
        gm = BayesianGaussianMixture(n_components=max_n_gaussians, max_iter=20000, n_init=1,
                                     verbose=51, verbose_interval=100, tol=1e-2,
                                     random_state=42, covariance_type='full')
        gm.fit(np.expand_dims(use.T[::subsampling], 1))  # gaussian mixture fit using N=floor(1/subsampling) cells
        preds = gm.predict(np.expand_dims(use.T, 1))  # gaussian mixture prediction for every cell

        # Shrinkage step: shrinking each point towards its belonging gaussian's mean
        preds_final = gm.means_[preds].T[0].T - (gm.means_[preds].T[0].T - use.T)/contraction_factor

        # backtransforming shrinked data to the original space, and setting each markers minimum equal to 0
        db.loc[:, column] = 2**(preds_final)-epsilon
        db.loc[:, column] -= db.loc[:, column].min()

        n_populations[column] = len(np.unique(preds))

        # saving checkpoint, in case something goes wrong mid-way, check for the last printed marker and perform preprocessing only on the remainin ones
        db.to_csv(base_folder+"quantized_dbs/"+area+"_quantized("+str(max_n_gaussians)+","+str(contraction_factor)+").csv", 
                  index=False)

        # print to file preprocessed markers list
        with open(folder+"/preprocessed_markers.txt", 'a') as f:
            print(column+" Done, with ", len(np.unique(preds)), "within-marker populations.", file=f)


    # robust standardization
    db = (db - db.median()) / (np.abs(db - db.mean()).mean())

    # save to .json file the number of gaussians (populations) per marker
    with open(base_folder+"quantized_dbs/"+area+"_n_populations.json", "w") as f:
        json.dump(n_populations, f)

    if populations_plot:
        # plot and save histogram for the number of gaussians (populations) per marker
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.hist(n_populations.values(), bins=np.arange(max_n_gaussians+1)+.5, density=False)
        ax.set_xlabel("number of identified gaussian components in a marker")
        ax.set_ylabel("number of markers with a given number of gaussian components")
        ax.grid()
        fig.tight_layout()
        fig.savefig(folder+area+"__gaussian_components_histogram.png", facecolor='w')
    
    # warn if there are saturated populations
    n_saturated_coolumns = np.where(n_populations == max_n_gaussians, 1, 0).sum()
    if n_saturated_coolumns:
        import warnings
        warnings.warn("Warning, some features saturated ("\
                      + str(n_saturated_coolumns)\
                      + " in total).\n The algorithm could be underperforming, you could try higher 'max_n_gaussians' parameter values to avoid this scenario",
                      RuntimeWarning)
    
    return db


# ----------------------------------------------------------------------------------------------------


def embed_dataset(db, nn=50, metric='euclidean', save_embed=True):

    """Perform the embedding on a 2D manifold of a pandas dataframe using the UMAP algorithm.
    
    
    \ndb (pandas DataFrame): the dataframe of which the embedding will be performed
    \nnn (integer): number of nearest neighbors to use during UMAP
    \nmetric (str, one of scipy-allowed distances): which metric to use during UMAP algorithm
    \nsave_embed: whether or not to save the resulting coordinates in the embedding space.
    
    """

    verboseprint("Starting UMAP embedding at:", datetime.now())
    # NOTE: You can't use spectral initialization with fully disconnected vertices or with datasets too big
    # Therefore, if spectral init fails, rerun using scaled pca init (scaled to 1e-4 standard deviation per dimension,
    # as suggested by Kobak and Berens 
    try:
        mapper = umap.UMAP(
            n_neighbors=nn,              # 50
            min_dist=0.0,                # 0.0
            n_components=2,              # 2
            random_state=42,             # 42
            verbose=True,                # TRUE
            low_memory=True,             # True
            init='spectral',             # spectral
            metric=metric,               # euclidean
        )

        embedding = mapper.fit_transform(db)
        if save_embed:
            embedding.dump(embedding_storage)  # save embedding coordinates for future uses

    except AttributeError:
        # scaled PCA initialization for UMAP due to huge db or other problems with spectral init
        print("Encountered problems with 'spectral' init, the recommended value.",
              "\nPreparing scaled PCA initialization and repeating UMAP embedding with this instead of spectral")

        # performing PCA on as many dimension as the UMAP 
        # to use its coordinate as a starting point for UMAP's embedding
        from sklearn.decomposition import PCA
        pca = PCA(n_components=mapper.n_components)
        inits = pca.fit_transform(db)

        # Kobak and Berens recommend scaling the PCA results 
        # so that the standard deviation for each dimension is 1e-4
        inits = inits / inits.std(axis=0) * 1e-4
        verboseprint('new embedding axis std:', inits.std(axis=0))

        # new trial with different init parameter
        mapper = umap.UMAP(
            n_neighbors=nn,              # 50
            min_dist=0.0,                # 0.0
            n_components=2,              # 2
            random_state=42,             # 42
            verbose=True,                # True
            low_memory=True,             # True
            init=inits,                  # inits
            metric=metric,               # euclidean
        )

        embedding = mapper.fit_transform(db)    
        if save_embed:
            embedding.dump(embedding_storage)  # save embedding coordinates for future uses

    verboseprint("Finished UMAP embedding at:", datetime.now())
    return embedding


# -------------------------------------------------------------------------

    
def markers_importance_per_cluster(use, clusters, compare_with='rest'):
    
    """Compute markers importance within each cluster.
    
    \nuse (pandas DataFrame): the dataframe to use for effect size and p-value computations, either original robustly scaled (suggested) or preprocessed db
    \nclusters (list or array-like): sequence of clusters labels (which cluster each point belongs to)
    \ncompare_with (str, either 'rest' or 'all'): whether or not to compare each cluster with the rest of the cells or with all the cells (including the cluster itself), the second option could be used to have a common reference for effect sizes and compare different markers effect sizes, but qould make the two samples for the ttest not indipendent.
    
    """
    
    unique_clusters = np.unique(clusters)


    # numerosity, mean, and std dev for each cluster
    n1 = use.groupby(clusters).count()
    m1 = use.groupby(clusters).mean()
    s1 = use.groupby(clusters).std()

    # numerosity, mean, and std dev for the whole sample
    n = len(use)
    m = use.mean()
    s = use.std()

    # numerosity, mean, and std dev for the rest of cells for each cluster
    n2 = n-n1
    m2 = (m*n-m1*n1)/n2
    s2 = ((n*s**2 - n1*(s1**2+(m1-m)**2))/n2 - (m2-m)**2 ) ** .5

    # output DataFrames where effect size and p-values will be stored
    res = m1.copy()    # effect size
    res_p = m1.copy()  # p-values



    if compare_with.lower() == 'all':
        for column in m1.columns:
            for ind in m1.index:
                # fast and vectorized version of t-test for p-values
                res_p.loc[ind, column] = st.ttest_ind_from_stats(m1.loc[ind, column],
                                                              s1.loc[ind, column],
                                                              n1.loc[ind, column],
                                                              m.loc[column],
                                                              s.loc[column],
                                                              n,
                                                              equal_var=False,
                                                              alternative='greater')[-1]
        # effect size computed with robust d by Vendekar et al.
        res = (-1)**(m1<m) * np.sqrt( (m1-m)**2 / ((n1+n)/n1*s1**2 + (n1+n)/n*s**2) )

    elif compare_with.lower() == 'rest':
        for column in m1.columns:
            for ind in m1.index:
                # fast and vectorized version of t-test for p-values
                res_p.loc[ind, column] = st.ttest_ind_from_stats(m1.loc[ind, column],
                                                              s1.loc[ind, column],
                                                              n1.loc[ind, column],
                                                              m2.loc[ind, column],
                                                              s2.loc[ind, column],
                                                              n2.loc[ind, column],
                                                              equal_var=False,
                                                              alternative='greater')[-1]
        # effect size computed with robust d by Vendekar et al.
        res = (-1)**(m2>m1) * np.sqrt( (m1-m2)**2 / ((n1+n2)/n1*s1**2 + (n1+n2)/n2*s2**2) )

    else:
        verboseprint("""Warning, it was unclear the data portion you wanted to compare results
              against, please use 'all' for using all the sample or 'rest' for using
              every cell outside the considered cluster.""")

    return res, res_p


# -------------------------------------------------------------------------


def most_important_effect_sizes(rs, rs_p, n, thrs_p=0.05, 
                                save_plot=False, path=base_folder+"cluster__expression.png"):
    
    """Horizontal bar plot with highest effect size markers for a given cluster
    
    \nrs (numpy array): a row for the 'res' output by 'markers_importance_per_cluster', represents each marker's effect size for a given cluster
    \nrs_p (numpy array): a row for the 'res_p' output by 'markers_importance_per_cluster', represents each marker's p-value for a given cluster
    \nn (integer): how many markers to plot, in descending order of effect size
    \nthrs_p (float between 0 and 1): change color for the horizontal bars based on whether or not the p-value is below this significance threshold
    \nsave (boolean): whether or not to save the resulting plot
    \npath (str): where to store the resulting plot (if saved)
    
    """

    # markers order based on descending effect size for top n markers                        
    order = np.argsort(rs)[::-1][:n]
    # consecutive differences
    diffs = np.ediff1d(rs[order])
    
    # 'last' is the last candidate to be in Tier 1 markers (within 0 and gap1)
    try:
        # 'last' = last positive effect size marker or second to last marker for top n effect sizes
        last = np.where(rs[order][:-2]<0.)[0][0]
        if last == 0:
            last = n-2
    except IndexError:
        last = n-2

    gap1 = diffs[:last].argmin()+1             # defining most expressed markers
    gap2 = diffs[gap1:last+1].argmin()+gap1+1  # defining possibly expressed markers

    
    # creating most expressed marker subtitle label
    lbl = "\nTier 1:"
    for _ in np.arange(gap1):
        lbl += " | "+find_names(db.columns[order][_])[0]+" "+str(rs[order][_])[:4]
    lbl += "\nTier 2:"
    for _ in np.arange(gap1, gap2):
        lbl += " | "+find_names(db.columns[order][_])[0]+" "+str(rs[order][_])[:4]

    # making effect size plot (a.k.a. most expressed markers)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.barh(y=np.arange(n)[::-1],
            width=rs[order],
            color=np.where(rs_p[order]<thrs_p, 'b', 'r'),
            tick_label=find_names(db.columns[order]))
    # plot some suggested effect thresholds
    ax.vlines(0.1,  -1, n, colors='k', linestyles='-', label='1st effect threshold <= 0.1 (None~small)')
    ax.vlines(0.25, -1, n, colors='k', linestyles='--', label='2nd effect threshold <= 0.25 (small~medium)')
    ax.vlines(0.4,  -1, n, colors='k', linestyles='-.', label='3rd effect threshold <= 0.4 (medium~big)')

    # plot Tier 1 and Tier 2 gaps positions
    ax.plot([-.01, max(0.4, max(rs))], [np.arange(n)[::-1][gap1]+.5, np.arange(n)[::-1][gap1]+.5],
            'm--', label='End of tier 1 markers')
    ax.plot([-.01, max(0.4, max(rs))], [np.arange(n)[::-1][gap2]+.5, np.arange(n)[::-1][gap2]+.5],
            'y--', label='End of tier 2 markers')

    ax.grid()
    ax.legend(loc='lower right')

    # manually adjust legend
    handles, labels = ax.legend().axes.get_legend_handles_labels()
    
    handles.append(Patch(facecolor='b', edgecolor='b'))
    labels.append("significant p value (<"+str(thrs_p)+")")
    handles.append(Patch(facecolor='r', edgecolor='r'))
    labels.append("NOT significant p value (>"+str(thrs_p)+")")
    ax.legend(handles, labels, loc='lower right')


    fig.suptitle("cluster "+str(i)+" composition")
    fig.tight_layout()

    if save_plot:
        fig.savefig(path, facecolor='w')
        
    return lbl


# PLOTS ----------------------------------------------------------------------------------------------


def plot_markers_on_embedding(db, original_db, embedding, real_pos,
                              save=False, path=folder+"markers/"):
    
    """Summary plot for all markers.
    
    \ndb (pandas.DataFrame): dataframe containing cells x markers values, of which we want to plot the summary for every marker
    \noriginal_db (pandas.DataFrame): original dataframe containing unprocessed markers values, of which we want to plot the summary for every marker
    \nembedding (2-dimensional arra-like): X,Y positions on the low dimensional embedding wherre to scatter plot data
    \nreal_pos (2-dimensional array-like): X,Y positions on the real space where to scatter plot data
    \nsave (boolean): whether or not to save the resulting plots
    \npath (str): path at which resulting plots should be stored, if saved

    """
    
    # plot marker 'd' on the embedding and the real space, showing both its 
    # unprocessed and its processed distributions
    for d in db.columns:
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        
        # embedding space
        a = ax[0][0].scatter(*embedding.T, c=db.loc[:, d], s=0.1, cmap='gist_ncar')
        ax[0][0].grid()
        ax[0][0].set_title('UMAP embedding')

        # real space
        ax[0][1].scatter(real_pos.iloc[:, 0], real_pos.iloc[:, 1], c=db.loc[:, d], s=0.1, cmap='gist_ncar')
        ax[0][1].grid()
        ax[0][1].set_title('real position')

        # processed distribution
        ax[1][0].hist(db.loc[:, d], bins=100, density=True)
        ax[1][0].grid()
        ax[1][0].set_title('distribution in the preprocessed log space')

        # original distribution
        ax[1][1].hist(original_db.loc[db.index, d], bins=100, density=True)
        ax[1][1].grid()
        ax[1][1].set_title('original raw distribution')

        # colorbar and final adjustments
        fig.colorbar(a, ax=ax[0][0]).set_label('log of marker expression', rotation=270, labelpad=15)
        fig.suptitle('marker: '+d)
        fig.tight_layout()
        
        if save:
            fig.savefig(path+d+".png", facecolor='w')
            
            

# ---------------------------------------------------------------------------------------------------


def plot_embedding_clusters(clusters_list, real_pos, embedding, clusterer,
                            save_plot=False, path=base_folder+"__clusters.png"):
    
    """Plot HDBSCAN clusters onto real space and UMAP embedding coordinates with appropriate colormap
    
    \nclusters_list (list or array-like): sequence of clusters labels (which cluster each point belongs to)
    \nreal_pos (2-dimensional array-like): X,Y positions on the real space where to scatter plot data
    \nembedding (2-dimensional arra-like): X,Y positions on the low dimensional embedding wherre to scatter plot data
    \nclusterer (fitted hdbscan.HDBSCAN clustering object): fitted HDBSCAN clustering object used to cluster data from the embedding space
    \nsave_plot (boolean): whether or not to save the resulting plots
    \npath (str): path where to store the resulting plots if saved
    
    """
    

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # scatterplot each single cluster with the respective label
    for cl in np.unique(clusters_list):
        mask = clusters_list == cl
        ax[0].scatter(*embedding[mask].T, s=0.1, color=custom_colormap.colors[(1+cl)%len(custom_colormap.colors)], label=str(cl))
        ax[1].scatter(real_pos.iloc[:, 0][mask], real_pos.iloc[:, 1][mask], s=0.1, color=custom_colormap.colors[(1+cl)%len(custom_colormap.colors)])

    # plot title
    fig.suptitle("\nHDBSCAN clustering (eps="+str(clusterer.cluster_selection_epsilon)\
                 +", "+str(100*clusterer.min_samples/len(clusters_list))[:6]+" % of cells for min_clusters)\nclusters: "+str(np.max(clusters_list)+1)+\
                 "   |   noise: "+str(np.where(clusters_list == -1, 1, 0).sum()))

    # show legend iff there are less than 150 total clusters, otherwise it would be unreadable
    if len(np.unique(clusters_list)) < 150:
        lgnd = ax[0].legend(loc='center', bbox_to_anchor=(1.15, 0.5),
                            ncol=4, fancybox=True, shadow=True,
                            title='clusters',
                            fontsize=11)

        # change the marker size manually for the whole legend
        for lh in lgnd.legendHandles:
            lh.set_sizes([200])

    fig.tight_layout()
    if save_plot:
        fig.savefig(path, facecolor='w')


# ------------------------------------------------------------------------


def whole_dataset_summary_plots(db, area, alpha=0.9, size=1., legend_size=200,
                                plot_with_legend=True, plot_without_legend=True,
                                plot_noise=True, save_plot=False, 
                                base_path=base_folder+""):
    
    """Plot a Maximum of 4 plots (2 with legend and 2 without legend) which summarize the main markers and their interpretative_column for each cluster

    \ndb (pandas.DataFrame): processed dataframe containing cells x markers values
    \narea (str): filename of the db
    \nalpha (float between 0 and 1): transparency for the plots
    \nsize (positive float): size for the dots of the scatter plots
    \nlegend_size (positive float): size for the dots in the legend
    \nplot_with_legend (bool): whether or not to make the 2 plots with legends (may be unreadable if too many clusters/too long labels are used)
    \nplot_without_legend (bool): whether or not to make the 2 plots without legends (useful if with legends the plot are unreadable, please notice that this parameter is not mandatory to be opposite to 'plot_with_legend')
    \plot_noise (bool) whether or not to plot the noise cluster if found by HDBSCAN
    \nsave_plot (boolean): whether or not to save the resulting plot
    \npath (str): where to store the resulting plot (if saved).
    
    NOTE: we assume that the global variables: custom_colormap and interpretative_column are properly defined
 
    """
    
    # SUMMARY PLOTS FOR THE ADDED COLUMNS, WITH/WITHOUT LEGEND 
    for ii, show in enumerate(['MARKERS', 'MARKERS', 
                               interpretative_column.upper(), interpretative_column.upper()]):
        if not plot_with_legend:
            if ii%2:
                continue

        if not plot_without_legend:
            if ii%2:
                continue

        verboseprint("Elaborating", show, "plot")

        # prepare color parameters
        if show == 'MARKERS':
            color = db.loc[:, "MainMarkers"]
        else:
            color = db.loc[:, interpretative_column]
        n_clust = len(np.unique(color))


        fig, ax = plt.subplots(1, 1, figsize=(19,9))

        # actual plot
        for i, column in enumerate(np.unique(color)):
            if column == '':
                if plot_noise:
                    ax.scatter(pos.iloc[:, 0][color==column], pos.iloc[:, 1][color==column],
                               alpha=alpha, color='black',
                               label=' Noise', s=size)
                else:
                    continue
            ax.scatter(pos.iloc[:, 0][color==column], pos.iloc[:, 1][color==column],
                       alpha=alpha, color=custom_colormap.colors[i%len(custom_colormap.colors)],
                       label=np.unique(color)[i], s=size)

        # plot legend iff ii is odd
        if ii%2:
            lgnd = ax.legend(loc='center left', bbox_to_anchor=(1., 0.5),
                             ncol=1+n_clust//31, fancybox=True, shadow=True,
                             title=show,
                             fontsize=11)

            #change the marker size manually for the whole legend
            for lh in lgnd.legendHandles:
                lh.set_sizes([legend_size])


        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Real space")
        ax.grid()


        fig.suptitle("Cell type classification - Area "+str(area)+" - "+show,
                     fontsize=20)

        fig.tight_layout()

        if save_plot:        
            if ii%2:
                fig.savefig(base_path+area+"__"+show+".png", facecolor='w')
            else:
                fig.savefig(base_path+area+"__"+show+"_nolegend.png", facecolor='w')


# ------------------------------------------------------------------------

    
def plot_cluster_spatial_location(lbl, embedding, pos, mask, 
                                  save_plot=False, 
                                  path=base_folder+"cluster__position.png"):

    """Plot cluster position in UMAP embedding and in real space, together with a summary of Tier1 and Tier2 markers.
    
    \nlbl (str): string produced by func containing Tier 1 and Tier 2 markers, which will be used as header for the plot 
    \nembedding (2-dimensional arra-like): X,Y positions on the low dimensional embedding wherre to scatter plot data
    \npos (2-dimensional array-like): X,Y positions on the real space where to scatter plot data    
    \nmask (array-like of boolean values): boolean values corresponding to whether or not a single cell is part of the considered cluster
    \nsave_plot (boolean): whether or not to save the resulting plot
    \npath (str): where to store the resulting plot (if saved)
    
    """
        
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # baseline light grey plot for whole sample
    ax[0].scatter(embedding.T[0], embedding.T[1], s=0.1, color='lightgrey', label='other')
    ax[1].scatter(pos.iloc[:, 0], pos.iloc[:, 1], s=0.1, color='lightgrey')

    # cluster plot with vivid red
    ax[0].scatter(embedding.T[0][mask], embedding.T[1][mask], s=3, color='r', label=str(i))
    ax[1].scatter(pos.iloc[:, 0][mask], pos.iloc[:, 1][mask], s=3, color='r')

    ax[0].set_title("UMAP embedding")
    ax[1].set_title("Real position")


    fig.suptitle("Cluster "+str(i)+": containing "+str(np.asarray(mask).sum())+" cells. Reporting marker name and effect size"+lbl)

    lgnd = ax[0].legend(loc='center', bbox_to_anchor=(1., 0.5),
                        ncol=1, fancybox=True, shadow=True,
                        title='clusters',
                        fontsize=11)

    #change the marker size manually for the whole legend
    for lh in lgnd.legendHandles:
        lh.set_sizes([200])

    fig.tight_layout()

    if save_plot:
        fig.savefig(path, facecolor='w')
    

# -------------------------------------------------------------------------


def monitoring_features_plot(original_db, mask, cluster_name, 
                             monitoring_features, palette=[(0.2, 0.5, 1.), (1., 0.2, 0.2)], 
                             save_plot=False, path=base_folder+"cluster__monitoring_featuresining.png"):
    
    """Plot Kernel Density Estimation to compare a cluster with whole dataset using some monitoring features.
    
    \noriginal_db (pandas.DataFrame): original and unprocessed dataframe containing cells x markers values
    \nmask (array-like of boolean values): boolean values corresponding to whether or not a single cell is part of the considered cluster
    \ncluster_name (str): label to use for the cluster name in the plot 
    \nmonitoring_features (list or array-like): list of features over which each cluster will be compared with whole sample using KDE plots
    \npalette (list of 2 rgb tuples): color palette to use for cluster or whole sample cells and KDE plots 
    \nsave_plot (boolean): whether or not to save the resulting plot
    \npath (str): where to store the resulting plot (if saved)
    
    """    
    
    # build 'sub_db' for monitoring features plot, such pandas DataFrame will be composed
    # of only the monitoring features, and will contain an extra column based on which 
    # cells belong to a cluster or to the whole sample (cluster cells will have duplicates)
    sub_db = pd.DataFrame({'expression': original_db.loc[:, monitoring_features].T.stack().values,
                           'marker': [a[0] for a in original_db.loc[:, monitoring_features].T.stack().index],
                           'cluster': list(np.where(mask, 
                                                    cluster_name,
                                                    'whole sample'))*len(monitoring_features)})
    sub_db2 = sub_db[sub_db.loc[:, 'cluster']!='whole sample']
    sub_db2.loc[:,'cluster']='whole sample'
    sub_db = pd.concat([sub_db,sub_db2])
    
    plt.figure(figsize=(18, 8))

    # # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
    g = sns.FacetGrid(data=sub_db, row='marker', hue='cluster', palette=palette,
                      sharey=False, legend_out=True, height=1, aspect=2*len(monitoring_features)-2)

    # then we add the densities kdeplots for each month
    g.map(sns.kdeplot, 'expression',
          bw_adjust=1, clip_on=False,
          fill=True, alpha=0.4, linewidth=2)

    # here we add a horizontal line for each plot
    g.map(plt.axhline, y=0, lw=5, clip_on=False)


    ft = 35 # bigger font size for some within plot headers
    
    # we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
    # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
    for ii, axis in enumerate(g.axes.flat):

        # <Specific for the paper>
        if area == 'DatasetCodex':
            axis.text(-1000, 0.0, monitoring_features[ii],
                      fontweight='bold', fontsize=ft, color=axis.lines[0].get_color())
            axis.text(8000, 0.0, 'effect size: '+str(res.loc[i, monitoring_features[ii]])[:6],
                      fontweight='bold', fontsize=ft, color=axis.lines[0].get_color())
        else:
            axis.text(-13, 0.0, monitoring_features[ii],
                      fontweight='bold', fontsize=ft, color=axis.lines[0].get_color())
            axis.text(100, 0.0, 'effect size: '+str(res.loc[i, monitoring_features[ii]])[:6],
                      fontweight='bold', fontsize=ft, color=axis.lines[0].get_color())
        axis.plot([sub_db2.groupby('marker').median().loc[monitoring_features[ii]],
                   sub_db2.groupby('marker').median().loc[monitoring_features[ii]]],
                   axis.get_ylim(), 'r--', linewidth=3)
        axis.plot([sub_db.groupby('marker').median().loc[monitoring_features[ii]],
                   sub_db.groupby('marker').median().loc[monitoring_features[ii]]], axis.get_ylim(), 'b--', linewidth=3)
        axis.set_ylabel("")

    # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.)

    # eventually we remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.setp(axis.get_xticklabels(), fontsize=ft, fontweight='bold')
    plt.xlabel('Marker Expression (a.u.)', fontweight='bold', fontsize=ft)
    g.fig.suptitle("", ha='right', fontsize=ft+5, fontweight=20)
    if area == 'DatasetCodex':
        plt.xlim(-1000, 10000)
    else:
        plt.xlim(-13, 150)
    plt.xticks(fontweight='bold', fontsize=ft//2)
    plt.legend()
    
    # manually adjust the legend labels
    handles, labels = [], []
    handles.append(Patch(facecolor='b', edgecolor='b', alpha=0.4))
    labels.append("whole sample")    
    handles.append(Patch(facecolor='r', edgecolor='r', alpha=0.4))
    labels.append(cluster_name)
    handles.append(Line2D([0], [0], linestyle='--', linewidth=5, color='b'))
    labels.append("marker median (whole sample)")
    handles.append(Line2D([0], [0], linestyle='--', linewidth=5, color='r'))
    labels.append("marker median (cluster)")
    
    plt.legend(handles, labels, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, len(monitoring_features)+1.01), fontsize=ft)
    
    
    if save_plot:
        plt.savefig(path, facecolor='w')       
        
        
        
        
        
        
        
        
custom_colormap = custom_colormap_for_many_clusters()


if not large_db_procedure:
    # FEATURES SELECTION, PREPROCESSING, and UMAP embedding

    # load reference file for columns: define subgroups, discarded markers, etc.
    reference = pd.read_csv(reference_file)
    print("loaded reference file")


    # creating directories to store plots/embedding
    for directory in [folder, folder+"positions/", folder+'expressions/', 
                      folder+"markers/", base_folder+"data",
                      base_folder+"quantized_dbs", base_folder+"embeddings"]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print("creating", directory)
        else:
            verboseprint("using pre existing", directory)

    # eventually load, previously preprocessed db
    if load_db:
        db = pd.read_csv(base_folder+"quantized_dbs/"+area+"_quantized("+str(max_n_gaussians)+","+str(contraction_factor)+").csv")

        # robust standardization
        db = (db-db.median())/(np.abs(db - db.mean()).mean())

        # select a specific marker subgroup, if required
        if markers_subgroup:
            for column in db.columns.values:
                verboseprint(column)
                marker = column.split(sep='_')[-1]

                # <SPECIFIC FOR THE PAPER>
                if marker == 'lambda':
                    ab = [a for a in reference.loc[:, correspondence_column] if marker in a][1]  # since kappasinelambda comes at 0
                else:
                    ab = [a for a in reference.loc[:, correspondence_column] if marker in a][0]

                # discard markers not in the specific subgroup
                to_discard = reference[reference.loc[:, correspondence_column] == ab].loc[: , markers_subgroup].iloc[0] == 0
                if to_discard:
                    db.drop(columns=column, inplace=True)
                    verboseprint("Dropping", column, "since this marker is not in", markers_subgroup, 'subgroup.')

        # measure the number of the remaining markers
        n_markers = len(db.columns)


    else:
        # <SPECIFIC FOR THE PAPER>
        if area == "DatasetCodex":
            verboseprint("Starting load of the data at:", datetime.now())
            original_db = pd.read_csv(base_folder+"data/"+area+".csv").iloc[:, 1:]

            # to have marker names in the same format as we are used, which is just marker name with no channel or separators
            new_names = {key:value for (key,value) in zip(original_db.columns, [a.split('.')[-1] for a in original_db.columns])}
            original_db.rename(columns=new_names, inplace=True)  # renames columns to just marker name

            # separate into db for markers and pos for spatial position
            db = original_db.iloc[:, 4:-3]
            pos = original_db.iloc[:, :2]
            print("data loaded")

        else:
            verboseprint("Starting load of the data at:", datetime.now())
            original_db = pd.read_csv(base_folder+"data/"+area+".csv")

            # to have marker names in the same format as we are used, which is just marker name with no channel or separators
            new_names = {key:value for (key,value) in zip(original_db.columns, [a.split('_')[-1] for a in original_db.columns])}

            # remove columns with identical names, keeping only the first one
            if len(set(new_names.values())) != len(new_names.values()):
                verboseprint("Same markers were found to have duplicates, keeping only first occurrence")
                names_counts = collections.Counter(new_names.values()).most_common()
                for (marker, count) in names_counts:
                    if count > 1:
                        verboseprint("dropping duplicates of", marker)
                        for column in [key for key, value in new_names.items() if value == marker][1:]:
                            original_db.drop(inplace=True, columns=column)
                            verboseprint(column, "dropped")

            # renames columns to just marker name
            original_db.rename(columns=new_names, inplace=True)  

            # separate into db for markers and pos for spatial position
            db = original_db.iloc[:, :-3]
            pos = original_db.iloc[:, -3:-1]
            print("data loaded")


        # discard unwanted markers, due to different possible reasons
        if perform_features_selection:
            db = features_selection(db, reference)

            # measure the number of the remaining markers
            n_markers = len(db.columns)

        # # perform Lognormal Shrinkage (LNS) from Dall'Olio et al. (2023) :https://doi.org/10.3390/e25020354
        # if perform_lognormal_shrinkage:
        #     db = lognormal_shrinkage(db, subsampling=4, max_n_gaussians=max_n_gaussians, 
        #                              contraction_factor=contraction_factor, populations_plot=True)

        else:
            verboseprint("Not performing data transformation.")



        # =============================================================================
        # UMAP EMBEDDING
        # =============================================================================


        if load_embed:
            print("Loading UMAP embedding...")
            # loading pre-made embedding file (if available)
            embedding = np.load(embedding_storage, allow_pickle=True)

        else:
            embedding = embed_dataset(db.iloc[:, :n_markers], save_embed=True, nn=nn, metric=metric)

else:
    # load only the nececessary information for clustering: i.e., umap's embedding coordinates, and the rest later
    print("Loading UMAP embedding...")
    embedding = np.load(embedding_storage, allow_pickle=True)

    
    # perform HDBSCAN clustering on c
    mindim = max(int(len(db)*0.00005), 10)  # minimum number of cells allowed to form a speparate cluster [0.05%,0.2%]
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mindim, min_samples=mindim,
                                cluster_selection_epsilon=0.1,  # between 0.08 and 0.12 usually
                                cluster_selection_method='eom')
    placeholder = clusterer.fit_predict(embedding)
    verboseprint('clustering large db')
    verboseprint(datetime.now())



    # only now load the whole db

    # load reference file for columns: define subgroups, discarded markers, etc.
    reference = pd.read_csv(reference_file)
    print("loaded reference file")


    # creating directories to store plots/embedding
    for directory in [folder, folder+"positions/", folder+'expressions/', folder+"markers/"]:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            print("creating", directory)
        else:
            verboseprint("using pre existing", directory)

    # load previously preprocessed db
    db = pd.read_csv(base_folder+"quantized_dbs/"+area+"_quantized("+str(max_n_gaussians)+","+str(contraction_factor)+").csv")

    # robust standardization
    db = (db-db.median())/(np.abs(db - db.mean()).mean())

    # select a specific marker subgroup, if required
    if markers_subgroup:
        for column in db.columns.values:
            verboseprint(column)
            marker = column.split(sep='_')[-1]

            # <SPECIFIC FOR THE PAPER>
            if marker == 'lambda':
                ab = [a for a in reference.loc[:, correspondence_column] if marker in a][1]  # since kappasinelambda comes at 0
            else:
                ab = [a for a in reference.loc[:, correspondence_column] if marker in a][0]

            # discard markers not in the specific subgroup
            to_discard = reference[reference.loc[:, correspondence_column] == ab].loc[: , markers_subgroup].iloc[0] == 0
            if to_discard:
                db.drop(columns=column, inplace=True)
                verboseprint("Dropping", column, "since this marker is not in", markers_subgroup, 'subgroup.')

    # measure the number of the remaining markers
    n_markers = len(db.columns)

    # assign clustering labels to db
    db['cluster'] = placeholder



plot_markers_on_embedding(db, original_db, embedding, pos, save=True, path=folder+"markers/")





# =============================================================================
# HDBSCAN clustering on UMAP embedding
# =============================================================================

if not large_db_procedure:
    if load_clusters:
        db['cluster'] = pd.read_csv(folder+area+"__clusters_labels.csv").loc[:, 'cluster']

    else:
        mindim = max(int(len(db)*0.00005), 10)  # minimum number of cells allowed to form a speparate cluster [0.05%,0.2%]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=mindim, min_samples=mindim,
                                    cluster_selection_epsilon=0.1,  # between 0.08 and 0.12 usually
                                    cluster_selection_method='eom')
        db['cluster'] = clusterer.fit_predict(embedding)


plot_embedding_clusters(db['cluster'], pos, embedding, clusterer, save_plot=True, path=folder+area+"__clusters.png")
    
verboseprint(datetime.now())









# optional reclustering procedure <Specific for the paper>
# tune another clustering with indipendent parameters on the biggest cluster
# (which probably needs to be furtherly splitted)

if not load_clusters:
    
    
    # Find biggest cluster as candidate for the reclustering procedure
    sizes = db.groupby('cluster').size()
    if sizes.max() > len(db)*0.1:
        sub_db = db[db['cluster'] == sizes.argmax()-1].copy()
        sub_pos = pos[db['cluster'] == sizes.argmax()-1].copy()
        candidate_reclustering_core = sizes.argmax()-1
        
    else:
        candidate_reclustering_core = -1

        
    # evaluate if biggest cluster is candidate for further splitting,
    # and in that case perform a reclustering of just the biggest cluster
    if candidate_reclustering_core != -1 and sizes.max() > len(db)/10:
        mask = db['cluster'] == candidate_reclustering_core
        sub_db = db[mask].copy()
        sub_embedding = embedding[mask].copy()
        sub_pos = pos[mask].copy()

        sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=mindim*10, min_samples=mindim*10,
                                        cluster_selection_epsilon=0.08,  # 0.08 for whole Lymph node, L1, CodeX
                                                                         # 0.11 for L2, L3, T1, T2
                                                                         # 0.15 for T3
                                        cluster_selection_method='leaf')
        sub_db['cluster'] = sub_clusterer.fit_predict(sub_embedding)

        plot_embedding_clusters(sub_db['cluster'], sub_pos, sub_embedding, sub_clusterer, save_plot=True, 
                                path=folder+area+"__clusters_reclustering_core_zoom.png")



        verboseprint(datetime.now())
        
        
        
# if optional reclustering procedure has been performed, 
# overwrite biggest cluster labels with new ones <Specific for the paper>

if not load_clusters:

    if candidate_reclustering_core != -1 and sizes.max() > len(db)/10:

        sub_db['cluster'] = np.where(sub_db['cluster'] < 1, sub_db['cluster'], sub_db['cluster']+max(db['cluster']))
        sub_db['cluster'] = np.where(sub_db['cluster'] == 0, candidate_reclustering_core, sub_db['cluster'])

        db.loc[sub_db.index, 'cluster'] = sub_db['cluster']

        plot_embedding_clusters(db['cluster'], pos, embedding, clusterer, save_plot=True, 
                                path=folder+area+"__clusters.png")

        verboseprint(datetime.now())
        
        
        
        
        
        
        
        
# =============================================================================
# ASSIGN CLUSTER SIGNIFICANCE
# =============================================================================



# choose on which db to calculate the metrics
select = "original"  # "original" suggested

if select.lower() == 'original':
    use = original_db.loc[:, db.columns[:n_markers]]  # use original db and
    use = (use-use.median())/(np.abs(use - use.mean()).mean())             # robustly standardize it
else:
    use = db.iloc[:, :n_markers]                      # use modified db

p_val_basic_threshold = 0.001  # p-value for a single test

# thresholds for pvalues (statistical significance) and for effect sizes (difference magnitude)
# using Bonferroni correction, most stringent one w.r.t. false positivies (e.g. falsely expressed markers)
thrs_p = p_val_basic_threshold/len(db.columns[:-1])/len(np.unique(db.loc[:, 'cluster']))
    
    
    
res, res_p = markers_importance_per_cluster(use, db.loc[:, 'cluster'], compare_with='rest')



db = add_main_markers_significance_columns(db, res, thrs_p=thrs_p)



whole_dataset_summary_plots(db, area, save_plot=True, base_path=folder)



# Find and order important markers (in our specific case, lineage defining markers)
# which will be used to monitor the nature of each cluster and compare it to the whole sample
monitoring_features = []

if area == "DatasetCodex":
    monitoring_features = use.columns
    
else:
    for column in use.columns:
        try:
            ab = [a for a in reference.loc[:, correspondence_column] if column in a][0]
            if reference[reference.loc[:, correspondence_column] == ab].loc[:, importance_column].iloc[0]:
                monitoring_features.append(column)
        except IndexError:
            verboseprint(column, "not present in reference file. Cannot establish if it is a monitoring feature")
            continue


# order markers alphabetically to always have the same set across different samples
monitoring_features = np.asarray(monitoring_features)[np.argsort(monitoring_features)]


ft = 35
plt.rcParams.update({'font.size': 15})

verboseprint("clusters:", np.unique(db.loc[:, 'cluster']))


for i in np.unique(db.loc[:, 'cluster']):
    verboseprint("producing results for cluster", i)

    # select cells within cluster i-th
    mask = db.loc[:, 'cluster'] == i  

    # select res and res_p corresponding to cluster i-th
    rs = np.asarray(res.loc[i, :])
    rs_p = np.asarray(res_p.loc[i, :])

    # <Specific for the paper>
    if area == 'DatasetCodex':
        n = np.min([14, len(rs)])  # consider a fixed maximum number of markers
    else:
        n = np.min([30, len(rs)])  # consider a fixed maximum number of markers

    # making effect sizes plot (ranking markers in descending order of effect size)
    # and building the 'lbl' variable, a string with Tier 1 (probably important markers)
    # and Tier 2 (possibly important markers)
    lbl = most_important_effect_sizes(rs, rs_p, n, thrs_p=thrs_p, save_plot=True, path=folder+"expressions/"+area+"__cluster_"+str(i)+"_expression.png")


    # making postion plot (where are cluster cells) and reporting most expressed markers
    plot_cluster_spatial_location(lbl, embedding, pos, mask, save_plot=True, path=folder+"positions/"+area+"__cluster_"+str(i)+"_position.png")


    # Monitoring features plot
    monitoring_features_plot(original_db, mask=mask, cluster_name='cluster '+str(i), monitoring_features=monitoring_features, save_plot=True, path=folder+"expressions/"+area+"__cluster_"+str(i)+"_monitoring_featuresining.png")
    
    
verboseprint(datetime.now())








# save results
if not load_clusters:
    # cluster labels
    db.loc[:, "cluster"].to_csv(folder+area+"__clusters_labels.csv")

    # cluster sizes
    sizes = db.groupby('cluster').size()
    sizes.to_excel(folder+area+"__clusters_sizes.xlsx", index_label='cluster')
    verboseprint(datetime.now())

    
# reset plot params to default
plt.rcParams.update({'font.size': old_fontsize})
