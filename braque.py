"""BRAQUE - Bayesian Reduction for Amplified Quantisation in Umap Embedding"""

import os
import json
import random

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





class BRAQUE():
    def __init__(self, original_db, pos, reference, correspondence_column, 
                 naming_column, interpretative_column, importance_column,
                 features_subgroup='', area='dataset', perform_features_selection=True, 
                 perform_lognormal_shrinkage=True, subsampling=1, max_n_gaussians=15,
                 contraction_factor=5., populations_plot=True, nn=50, metric='euclidean', 
                 HDBSCAN_merging_parameter=0.1, reclustering_step=False, 
                 p_val_basic_threshold = 0.05, load_embed=False, load_db=False,
                 load_clusters=False, base_folder='./', save_plots=True, verbose=False):
        
        
        """BRAQUE object to perform the pipeline reported in Dall'Olio et al. https://doi.org/10.3390/e25020354.  
        
        \noriginal_db (pandas DataFrame, n_cells x n_features shaped): data on which to perform the analysis, with units (e.g., cells) on rows and features (e.g. features) on columns. This variable will remain untouched, and will be used at the end for statistical comparisons.
        \npos (pandas DataFrame, n_cells x 2 shaped): spatial positional features for your units (e.g. x,y columns for real space coordinates).
        \nreference (pandas DataFrame, n_features x n_properties shaped): DataFrame where every row must correspond to a different feature and every column should provide a different property of such feature, few columns are mandatory, like corresponding_column, naming_column, interpretative_column and importance_column, see below for further details.
        \ncorrespondence_column (str): header of the reference file column which contains features names, may contain multiple variants in the format "variant1/variant2/.../variantN".
        \nnaming_column (str): header of the reference file column which contains features names that shall be used in plots/results.
        \ninterpretative_column (str): header of the reference file column which contains features associated property.
        \nimportance_column (str): header of the reference file column which contains 1 for important features that should be used for summary plot.
        \nfeatures_subgroup (str): optional header of the reference file column which might opionally be used to keep only a subset of features. if used shall be a 0/1 coded column, with 1 for keeping the feature at that specific row, or 0 to exlude it. use an empty string ("") to avoid such subselection.
        \narea (str): string that will be used for naming the folders and plots correspond to the current dataset/analysis.
        \nperform_feature_selection (boolean): whether or not to perform features selection.
        \nperform_lognormal_shrinkage (boolean): whether or not to perform lognormal shrinkage (preprocessing from Dall'Olio et al.).
        \nsubsampling (integer, between 1 and len(db)): subsampling parameter, take 1 cell every N. in order to speed up gaussian mixture fitting procedure
    \nmax_n_gaussians (positive integer, >=2): maximum number of fittable lognormal distributions for a single feature, keep in mind that the higher the slower and more precise the algorithm. To tune follow guidelines from Dall'Olio et al.
    \ncontraction_factor (positive float, >1.): each gaussian in the log2 space is contracted by this factor to better separate candidate subpopulations. To tune follow guidelines from Dall'Olio et al.
    \npopulations_plot (boolean): whether or not to plot the final summary about number of candidates subpopulations for each feature, useful to tune max_n_gaussians.
        \nnn (integer): number of nearest neighbors to use during UMAP
        \nmetric (str, one of scipy-allowed distances): which metric to use during UMAP algorithm
        \nHDBSCAN_merging_parameter (float, non-negative): corresponds to 'cluster_selection_epsilon' of the HDBSCAN algorithm.
        \nreclustering_step (boolean): whether or not to perform a second HDBSCAN clustering on the biggest cluster to unpack eventual superclusters that may form in immunofluorescence context, do not use if you are not sure.
        \np_val_basic_threshold (float, between 0 and 1 excluded): which interpretative_column threshold should be adopted for a single test, such threshold will be bonferroni corrected for multiple tests scenarios.
        \nload_embed (boolean): whether or not to load precomputed embedding from the /embeddings/ subfolder.
        \nload_db (boolean): whether or not to load precomputed processed db from the /quantized_dbs/ subfolder. 
        \nload_clusters (boolean): whether or not to load precomputed clusters from the /results/area/ subfolder.
        \nbase_folder (str): root folder from which the analysis tree will start and be performed, within this folder plots and results will e stored in appropriate subfolders.
        \nsave_plots (boolean): whether or not to store the produced plots.
        \nverbose (boolean): whether or not to obtain a more verbose output.
        
        
        """
        
        

        # select if you wish a verbose print or not
        self.verboseprint = print if verbose else lambda *a, **k: None
        self.verboseprint(datetime.now())


        # dataset
        self.area = area
        
        # reference file
        self.reference = reference  # reference file
        self.correspondence_column = correspondence_column  # header of the reference file column which contains features names, may contain variants
        self.naming_column = naming_column  # header of the reference file column which contains features names that shall be used in plots/results
        self.interpretative_column = interpretative_column  # header of the reference file column which contains features associated propery
        self.importance_column = importance_column  # header of the reference file column which contains 1 for important features that should be used for summary plot

        self.features_subgroup = features_subgroup    # header of the reference file column which might opionally be used to keep only a subset of features. if used shall be a 0/1 coded column, with 1 for keeping the feature at that specific row, or 0 to exlude it. use an empty string ('') to avoid such subselection


        # PREPROCESSING PARAMETERS
        self.perform_features_selection = perform_features_selection

        # BayesianGaussianMixture
        self.perform_lognormal_shrinkage = perform_lognormal_shrinkage  # wether or not perform LNS (Lognormal Shrinkage preprocessing from Dall'Olio et al. https://doi.org/10.3390/e25020354
        self.max_n_gaussians = max_n_gaussians  # 15 for cores and 20 for limphnode
        self.subsampling = subsampling  # subsampling parameter to fasten lognormalshrinkage, 1 gives the slowest and most accurate results, the higher the faster but with less accurate outcomes
        self.contraction_factor = contraction_factor  # between 2 and 10, suggested 5, is affected by the chosen base for the logarithm
        self.populations_plot = populations_plot

        # UMAP
        self.nn = nn # 50, but should not brutally affect varying in hte range 30~100
        self.metric = metric  # euclidean if no 
        
        # HDBSCAN
        self.HDBSCAN_merging_parameter = HDBSCAN_merging_parameter
        self.reclustering_step = reclustering_step
        
        # statistical analysis
        self.p_val_basic_threshold = p_val_basic_threshold

        # load premade steps if needed
        self.load_embed = load_embed
        self.load_db = load_db 
        self.load_clusters = load_clusters
        
        if load_db or load_embed or load_clusters:
            self.perform_features_selection = False
            self.perform_lognormal_shrinkage = False


        # saving directories for resulting plots
        self.base_folder = base_folder 
        self.folder = base_folder+"results/"+area+features_subgroup+"/"
        self.embedding_storage = base_folder+"embeddings/"+area+features_subgroup

        self.save_plots = save_plots
        


        # creating directories to store plots/embedding
        for directory in [self.folder, self.folder+"positions/", self.folder+'expressions/',
                          self.folder+"features/", base_folder+"data",
                          base_folder+"quantized_dbs", base_folder+"embeddings"]:
            if not os.path.isdir(directory):
                os.makedirs(directory)
                print("creating", directory)
            else:
                self.verboseprint("using pre existing", directory)

       
        # eventually load, previously preprocessed db
        if self.load_db:
            print("Loading preprocessed db...")
            db = pd.read_csv(base_folder+"quantized_dbs/"+area+"_quantized("+str(max_n_gaussians)+","+str(contraction_factor)+").csv")

            # robust standardization
            db = (db-db.median())/(np.abs(db - db.mean()).mean())

            # select a specific feature subgroup, if required
            if self.features_subgroup:
                for column in db.columns.values:
                    self.verboseprint(column)
                    corresponding_feature = column.split(sep='_')[-1]

                    # <SPECIFIC FOR THE PAPER>
                    if corresponding_feature == 'lambda':
                        feature = [a for a in self.reference.loc[:, self.correspondence_column] if corresponding_feature in a][1]  # since kappasinelambda comes at 0
                    else:
                        feature = [a for a in self.reference.loc[:, self.correspondence_column] if corresponding_feature in a][0]

                    # discard features not in the specific subgroup
                    to_discard = self.reference[self.reference.loc[:, self.correspondence_column] == feature ].loc[: , self.features_subgroup].iloc[0] == 0
                    if to_discard:
                        db.drop(columns=column, inplace=True)
                        self.verboseprint("Dropping", column, "since this feature is not in", self.features_subgroup, 'subgroup.')


        else:
            self.verboseprint("Starting load of the data at:", datetime.now())

            
            db = original_db.copy()

            # select a specific feature subgroup, if required
            if self.features_subgroup:
                for column in db.columns.values:
                    self.verboseprint(column)
                    corresponding_feature = column.split(sep='_')[-1]

                    # <SPECIFIC FOR THE PAPER>
                    if corresponding_feature == 'lambda':
                        feature = [a for a in self.reference.loc[:, self.correspondence_column] if corresponding_feature in a][1]  # since kappasinelambda comes at 0
                    else:
                        feature = [a for a in self.reference.loc[:, self.correspondence_column] if corresponding_feature in a][0]

                    # discard features not in the specific subgroup
                    to_discard = self.reference[self.reference.loc[:, self.correspondence_column] == feature ].loc[: , self.features_subgroup].iloc[0] == 0
                    if to_discard:
                        db.drop(columns=column, inplace=True)
                        self.verboseprint("Dropping", column, "since this feature is not in", self.features_subgroup, 'subgroup.')
          
            print("data loaded")

        # assign dataset-related attributes
        self.original_db = original_db
        self.db = db
        self.pos = pos
        self.n_features = len(db.columns)
        # built custom colormap
        self.custom_colormap_for_many_clusters()
        
        # use an estimate to roguhly have visible plots with any numerosity
        self.dot_size = 11-np.log2(len(db))
        if self.dot_size < 1:
            self.dot_size = 10/np.log2(len(db))
            

        if self.load_embed:
            print("Loading UMAP embedding...")
            # loading pre-made embedding file (if available)
            self.embedding = np.load(self.embedding_storage, allow_pickle=True)

        
        if self.load_clusters:
            print("Loading precomputed clusters...")
            self.clusters = pd.read_csv(self.folder+self.area+"__clusters_labels.csv").loc[:, 'cluster']



    # UTILS -----------------------------------------------------------

    def custom_colormap_for_many_clusters(self, n_clusters=None, random_seed=42, 
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
                self.verboseprint("removed color", i)

        # keep only n_clusters different colors in the final colormap
        for i in output_colors[:n_clusters]:
            colors_rgb_list.append(mcolors.hex2color(colors_list[i]))

        self.custom_colormap = mcolors.ListedColormap(colors_rgb_list)


    # ----------------------------------------------------------------------


    def find_names(self, features_names):

        """Substitutes columns names with pre-defined standard names contained in reference file.

        \nfeatures_names (string or list/array-like): either single string to convert or list of strings to convert to standard name.
        It is important for these values to exactly correspond to a value of the reference file corresponding column.

        """

        official_features_names = []

        if isinstance(features_names, str):  
            # only 1 iteration to perform, otherwise we cicle over the single characters within the string
            try:
                feature = [a for a in self.reference.loc[:, self.correspondence_column] if features_names in a][0]
                official_features_names.append(self.reference[self.reference.loc[:, self.correspondence_column] == feature].loc[:, self.naming_column].iloc[0]) 
            except IndexError:
                official_features_names.append(features_names)
            return np.asarray(official_features_names)

        else:
            for mn in features_names:
                try:
                    feature = [a for a in self.reference.loc[:, self.correspondence_column] if mn in a][0]
                    official_features_names.append(self.reference[self.reference.loc[:, self.correspondence_column] == feature].loc[:, self.naming_column].iloc[0]) 
                except IndexError:
                    official_features_names.append(mn)
            return np.asarray(official_features_names)


    # -------------------------------------------------------------------------


    def add_main_features_interpretative_columns(self, find_n=3, undef_thr=0.):

        """Add inplace to input pandas dataframe the column 'MainFeatures' and a column with their interpretation.

        \nfind_n (positive integer): how many main features to find, at most
        \nundef_thr (non-negative float): threshold below which an effect size is never considered relevant

        """

        prevalent_population = []
        features = []
        self.db[self.interpretative_column] =  ""
        self.db['MainFeatures'] = ""
        unique_clusters = np.unique(self.clusters)

        # using threshold suggested in robust Cohen's d by Vandekar et Al (2020)
        # undef_thr = 0.1

        for _ in range(1, find_n+1):
            # find features names for 'find_n' features with maximum effect size
            maxfeatures = [mrkr.split(sep='_')[-1] for mrkr in self.res.iloc[:, :].apply(lambda row: row.nlargest(_).index[-1], axis=1)]
            self.verboseprint("\n\n------------------------------------------\n",
                              _, "most expressed feature per cluster:\n")

            for i in unique_clusters:
                # if feature has larger effect size than 'udenf_thr' it will appear in Minfeatures column, 
                # otherwise it will be considered not expressed enough and therefore ignored
                if np.max(self.res.loc[i]) > undef_thr:
                    # add the effect size value in the label
                    level = self.res.loc[i, self.res.iloc[:, :].apply(lambda row: row.nlargest(_).index[-1], axis=1)[i]]
                    if level < undef_thr:
                        continue  # check, this could give error if no significant feature is found
                                  # and maxfeature/prevalent population result in being shorter

                    maxfeature = maxfeatures[i-min(unique_clusters)]


                    try:
                        feature = [a for a in self.reference.loc[:, self.correspondence_column] if maxfeature in a][0]
                        pop = self.reference[self.reference.loc[:, self.correspondence_column] == feature ].loc[:, self.interpretative_column].iloc[0]
                        features.append(self.find_names(maxfeature)[0])  # using Name instead of feature 

                        # label features with no clear correspondence as unclear
                        if pd.isna(pop):
                            prevalent_population.append("unclear")
                            self.verboseprint("cluster", i, "unclear nan", maxfeature)
                        else:
                            prevalent_population.append(pop)
                            self.verboseprint("cluster", i, maxfeature, pop)
                    except IndexError:
                        prevalent_population.append("unclear")
                        self.verboseprint("cluster", i, "unclear", maxfeature)

                    # for larger values of 'find_n' insert some newlines, in this case every 6 features names
                    if find_n > 6:
                        features[-1] += ':'+str(level)[:3]
                        if _%7 == 6:
                            features[-1] += '\n'
                else:
                    prevalent_population.append("unclear")
                    self.verboseprint("cluster", i, "undefined")

                # write MainFeatures column and interpretation column
                if features != []:
                    mask = self.db[self.clusters==i].index.values
                    self.db.loc[mask, self.interpretative_column] += " | "+prevalent_population[-1]
                    self.db.loc[mask, "MainFeatures"] += " | "+features[-1]
                else:
                    mask = self.db[self.clusters==i].index.values
                    self.db.loc[mask, self.interpretative_column] += " | "+prevalent_population[-1]
                    self.db.loc[mask, "MainFeatures"] += " | "


        # nullify noise labels
        mask = self.clusters == -1  # do not plot noise (a.k.a. removed cells)
        self.db.loc[:, "MainFeatures"][mask] = ''
        self.db.loc[:, self.interpretative_column][mask] = ''

        self.verboseprint("\n\n")
        for i in np.unique(self.clusters):
            self.verboseprint(i, self.db.loc[:, "MainFeatures"][self.clusters==i].iloc[0])




    # PIPELINE -------------------------------------------------------------------------------------


    def features_selection(self, drop_unclear=True, 
                           drop_missing=True, to_drop = ['IGNORE'], 
                           special_keeps=[]):

        """Perform features selection over a dataframe, given a reference file on which column to keep/discard.        

        \ndrop_unclear (boolean): whether to drop features with no corrispondence in the reference file
        \ndrop_missing (boolean): whether to drop features with missing interpretative_column
        \nto_drop (list or array-like): drop features whose column named 'interpretative_column' value, in the reference file, is part of this list
        \nspecial_keeps (list or array-like): features whose name is in this lisy will be kept anyway if they have at least a interpretative_column and it's != IGNORE

        """

        for column in self.db.columns.values:
            feature = column.split(sep='_')[-1]
            try:
                # <SPECIFIC FOR THE PAPER>
                if feature == 'lambda':
                    feature = [a for a in self.reference.loc[:, self.correspondence_column] if feature in a][1]  # since kappasinelambda comes at 0
                else:
                    feature = [a for a in self.reference.loc[:, self.correspondence_column] if feature in a][0]

                if self.features_subgroup:
                    # discard features not in the specific subgroup
                    to_discard = self.reference[self.reference.loc[:, self.correspondence_column] == feature].loc[: , self.features_subgroup].iloc[0] == 0
                    if to_discard:
                        self.db.drop(columns=column, inplace=True)
                        self.verboseprint("Dropping", column, "since this feature is not in", self.features_subgroup, 'subgroup.')
                        continue

                # drop unwanted populations, and/or missing/unclear ones
                population = self.reference[self.reference.loc[:, self.correspondence_column] == feature].loc[:, self.interpretative_column].iloc[0]

                # missing
                if drop_missing and pd.isna(population):
                    self.db.drop(columns=column, inplace=True)
                    self.verboseprint("Dropping", column, "due to 'no specific "+self.interpretative_column+".")

                # unwanted
                else:
                    for drop_candidate in to_drop:
                        if drop_candidate in population:
                            self.db.drop(columns=column, inplace=True)
                            self.verboseprint("Dropping", column, "due to unwanted "+self.interpretative_column+":", drop_candidate)

            # unclear
            except IndexError:
                if drop_unclear:
                    self.db.drop(columns=column, inplace=True)
                    self.verboseprint("Dropping", column, "due to 'unknown "+self.interpretative_column+"'.")

        self.n_features = len(self.db.columns)


    # ----------------------------------------------------------------------------------------------------


    def lognormal_shrinkage(self, subsampling=1, max_n_gaussians=20, log_transform=True,
                            contraction_factor=5., populations_plot=False):

        """Perform Lognormal Shrinkage preprocessing over a pandas datafame.

        \nsubsampling (positive integer, between 1 and len(db)): subsampling parameter, take 1 cell every N. in order to speed up gaussian mixture fitting procedure
        \nmax_n_gaussians (positive integer, >=2): maximum number of fittable lognormal distributions for a single feature, keep in mind that the higher the slower and more precise the algorithm. To tune follow guidelines from Dall'Olio et al.
        \nlog_transform (boolean): whether to do a lognormal mixture (suggested if data are not very multi gaussian) or a gaussian mixture.
        \ncontraction_factor (positive float, >1.): each gaussian in the log2 space is contracted by this factor to better separate candidate subpopulations. To tune follow guidelines from Dall'Olio et al.
        \npopulations_plot (boolean): whether or not to plot the final summary about number of candidates subpopulations for each feature, useful to tune max_n_gaussians.

        """
        
        old_fontsize = plt.rcParams['font.size']
        new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
        plt.rcParams.update({'font.size': new_fontsize})

        from sklearn.mixture import BayesianGaussianMixture

        self.verboseprint("performing Lognormal Shrinkage transformation")

        # dropping constant feature, to avoid erros and because they are useless
        for column in self.db.columns[np.where(self.db.var() == 0)[0]]:
            self.db.drop(columns=column, inplace=True)
            self.n_features -= 1
            self.verboseprint("Dropping", column, "due to Variance = 0. \n(It is not possible to fit Bayesian Gaussian Mixture on it, the signal is constant!)")

        # robust scaling, needed to use the same mixture parameters for all the features, which could otherwise strongly differ in range, eìoutlier percentage, etc.
        self.db = self.db/(np.abs(self.db - self.db.mean()).mean())
        
        # optionally print features to check their order/number/correctness
        self.verboseprint("list of genes to process")
        for i, column in enumerate(self.db.columns):
            self.verboseprint(i+1, column)

        if log_transform:
            # add small positive constant to the self.db, in order to avoid log(0) related errors
            epsilon = self.db[self.db>0.].min().min()
            logdb = np.log2(self.db+epsilon)

        with open(self.folder+"/preprocessed_features.txt", 'w') as f:
                print("list of features which completed the lognormal shrinkage preprocessing step", file=f)

        n_populations = {}  # to store number of gaussians per feature information
        for i, column in enumerate(self.db.columns):
            self.verboseprint(i+1, column)

            if log_transform:
                use = logdb.loc[:, column]
            else:
                use = self.db.loc[:, column]
                
            # Fitting a Bayesian Gaussian Mixture, with maximum number of gaussians = max_n_gaussian, over each feature separately
            gm = BayesianGaussianMixture(n_components=max_n_gaussians, max_iter=20000, n_init=1,
                                         verbose=51, verbose_interval=100, tol=1e-2,
                                         random_state=42, covariance_type='full')
            gm.fit(np.expand_dims(use.T[::subsampling], 1))  # gaussian mixture fit using N=floor(1/subsampling) cells
            preds = gm.predict(np.expand_dims(use.T, 1))  # gaussian mixture prediction for every cell

            # Shrinkage step: shrinking each point towards its belonging gaussian's mean
            preds_final = gm.means_[preds].T[0].T - (gm.means_[preds].T[0].T - use.T)/contraction_factor

            if log_transform:
                # backtransforming shrinked data to the original space, and setting each features minimum equal to 0
                self.db.loc[:, column] = 2**(preds_final)-epsilon
            self.db.loc[:, column] -= self.db.loc[:, column].min()

            n_populations[column] = len(np.unique(preds))

            # saving checkpoint, in case something goes wrong mid-way, check for the last printed feature and perform preprocessing only on the remainin ones
            self.db.to_csv(self.base_folder+"quantized_dbs/"+self.area+"_quantized("+str(max_n_gaussians)+","+str(contraction_factor)+").csv", 
                      index=False)

            # print to file preprocessed features list
            with open(self.folder+"/preprocessed_features.txt", 'a') as f:
                print(column+" Done, with ", len(np.unique(preds)), "within-feature populations.", file=f)


        # robust standardization
        self.db = (self.db - self.db.median()) / (np.abs(self.db - self.db.mean()).mean())

        # save to .json file the number of gaussians (populations) per feature
        with open(self.base_folder+"quantized_dbs/"+self.area+"_n_populations.json", "w") as f:
            json.dump(n_populations, f)

        if populations_plot:
            # plot and save histogram for the number of gaussians (populations) per feature
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.hist(n_populations.values(), bins=np.arange(max_n_gaussians+1)+.5, density=False)
            ax.set_xlabel("number of identified gaussian components in a feature")
            ax.set_ylabel("number of features with a given number of gaussian components")
            ax.grid()
            fig.tight_layout()
            if self.save_plots:
                fig.savefig(self.folder+self.area+"__gaussian_components_histogram.png", facecolor='w')
                
        # reset plot params to default
        plt.rcParams.update({'font.size': old_fontsize})
    

        # warn if there are saturated populations
        n_saturated_coolumns = np.where(n_populations == max_n_gaussians, 1, 0).sum()
        if n_saturated_coolumns:
            import warnings
            warnings.warn("Warning, some features saturated ("\
                          + str(n_saturated_coolumns)\
                          + " in total).\n The algorithm could be underperforming, you could try higher 'max_n_gaussians' parameter values to avoid this scenario",
                          RuntimeWarning)

        


    # -----------------------------------------------------------------


    def embed_dataset(self, nn=50, metric='euclidean', save_embed=True):

        """Perform the embedding on a 2D manifold of a pandas dataframe using the UMAP algorithm.

        \nnn (integer): number of nearest neighbors to use during UMAP
        \nmetric (str, one of scipy-allowed distances): which metric to use during UMAP algorithm
        \nsave_embed (boolean): whether or not to save the resulting coordinates in the embedding space.

        """

        self.verboseprint("Starting UMAP embedding at:", datetime.now())
        # NOTE: You can't use spectral initialization with fully disconnected vertices or with datasets too big
        # Therefore, if spectral init fails, rerun using scaled pca init (scaled to 1e-4 standard deviation per dimension,
        # as suggested by Kobak and Berens 
        
        if len(self.db.columns) != self.n_features:
            import warnings
            warnings.warn("Warning, the number of columns in preprocessed 'db' is "+str(len(self.db.columns))+", and does not match the number of expected features 'n_features' which is "+str(self.n_features)+". Please check that the 'db' attribute has not been unwillingly edited. To get rid of this warning run 'self.n_features = len(self.db.columns)' if you edited the 'db' attribute on purpose", RunetimeWarning)
        
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

            embedding = mapper.fit_transform(self.db)
            if save_embed:
                embedding.dump(self.embedding_storage)  # save embedding coordinates for future uses

        except AttributeError:
            # scaled PCA initialization for UMAP due to huge db or other problems with spectral init
            print("Encountered problems with 'spectral' init, the recommended value.",
                  "\nPreparing scaled PCA initialization and repeating UMAP embedding with this instead of spectral")

            # performing PCA on as many dimension as the UMAP 
            # to use its coordinate as a starting point for UMAP's embedding
            from sklearn.decomposition import PCA
            pca = PCA(n_components=mapper.n_components)
            inits = pca.fit_transform(self.db)

            # Kobak and Berens recommend scaling the PCA results 
            # so that the standard deviation for each dimension is 1e-4
            inits = inits / inits.std(axis=0) * 1e-4
            self.verboseprint('new embedding axis std:', inits.std(axis=0))

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

            embedding = mapper.fit_transform(self.db)    
            if save_embed:
                embedding.dump(self.embedding_storage)  # save embedding coordinates for future uses

        self.verboseprint("Finished UMAP embedding at:", datetime.now())
        self.embedding = embedding
        self.mapper = mapper


    # -------------------------------------------------------------------------


    def features_importance_per_cluster(self, select='original', compare_with='rest'):

        """Compute features importance within each cluster.

        \nselect (str): select which dataframe to use for effect size and p-value computations, either original robustly scaled ('original', suggested) or preprocessed db (any other string)
        \ncompare_with (str, either 'rest' or 'all'): whether or not to compare each cluster with the rest of the cells or with all the cells (including the cluster itself), the second option could be used to have a common reference for effect sizes and compare different features effect sizes, but qould make the two samples for the ttest not indipendent.

        """

        
        if len(self.db.columns) != self.n_features:
            import warnings
            warnings.warn("Warning, the number of columns in preprocessed 'db' is "+str(len(self.db.columns))+", and does not match the number of expected features 'n_features' which is "+str(self.n_features)+". Please check that the 'db' attribute has not been unwillingly edited. To get rid of this warning run 'self.n_features = len(self.db.columns)' if you edited the 'db' attribute on purpose", RunetimeWarning)
            
        if select.lower() == 'original':
            use = self.original_db.loc[:, self.db.columns[:self.n_features]]  # use original db and
            use = (use-use.median())/(np.abs(use - use.mean()).mean())             # robustly standardize it
        else:
            use = self.db.iloc[:, :self.n_features]                      # use modified db


        # numerosity, mean, and std dev for each cluster
        n1 = use.groupby(self.clusters).count()
        m1 = use.groupby(self.clusters).mean()
        s1 = use.groupby(self.clusters).std()

        # numerosity, mean, and std dev for the whole sample
        n = len(use)
        m = use.mean()
        s = use.std()

        # numerosity, mean, and std dev for the rest of cells for each cluster
        n2 = n-n1
        m2 = (m*n-m1*n1)/n2
        s2 = ((n*s**2 - n1*(s1**2+(m1-m)**2))/n2 - (m2-m)**2 ) ** .5

        # output DataFrames where effect size and p-values will be stored
        self.res = m1.copy()    # effect size
        self.res_p = m1.copy()  # p-values



        if compare_with.lower() == 'all':
            for column in m1.columns:
                for ind in m1.index:
                    # fast and vectorized version of t-test for p-values
                    self.res_p.loc[ind, column] = st.ttest_ind_from_stats(m1.loc[ind, column],
                                                                  s1.loc[ind, column],
                                                                  n1.loc[ind, column],
                                                                  m.loc[column],
                                                                  s.loc[column],
                                                                  n,
                                                                  equal_var=False,
                                                                  alternative='greater')[-1]
            # effect size computed with robust d by Vendekar et al.
            self.res = (-1)**(m1<m) * np.sqrt( (m1-m)**2 / ((n1+n)/n1*s1**2 + (n1+n)/n*s**2) )

        elif compare_with.lower() == 'rest':
            for column in m1.columns:
                for ind in m1.index:
                    # fast and vectorized version of t-test for p-values
                    self.res_p.loc[ind, column] = st.ttest_ind_from_stats(m1.loc[ind, column],
                                                                  s1.loc[ind, column],
                                                                  n1.loc[ind, column],
                                                                  m2.loc[ind, column],
                                                                  s2.loc[ind, column],
                                                                  n2.loc[ind, column],
                                                                  equal_var=False,
                                                                  alternative='greater')[-1]
            # effect size computed with robust d by Vendekar et al.
            self.res = (-1)**(m2>m1) * np.sqrt( (m1-m2)**2 / ((n1+n2)/n1*s1**2 + (n1+n2)/n2*s2**2) )

        else:
            self.verboseprint("""Warning, it was unclear the data portion you wanted to compare results
                  against, please use 'all' for using all the sample or 'rest' for using
                  every cell outside the considered cluster.""")


    # -------------------------------------------------------------------------


    def most_important_cluster_effect_sizes(self, rs, rs_p, n=10, cluster_number=-2, path="./cluster__expression.png"):

        """Horizontal bar plot with highest effect size features for a given cluster

        \nrs (numpy array): a row for the 'res' output by 'features_importance_per_cluster', represents each feature's effect size for a given cluster
        \nrs_p (numpy array): a row for the 'res_p' output by 'features_importance_per_cluster', represents each feature's p-value for a given cluster
        \nn (integer): how many features to plot, in descending order of effect size
        \ncluster_number (integer, non-negative): number to use as header for cluster name, useful to distinguish multiple clusters.
        \npath (str): where to store the resulting plot (if saved)

        """

        old_fontsize = plt.rcParams['font.size']
        new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
        plt.rcParams.update({'font.size': new_fontsize})

        
        # features order based on descending effect size for top n features                        
        order = np.argsort(rs)[::-1][:n]
        # consecutive differences
        diffs = np.ediff1d(rs[order])

        # 'last' is the last candidate to be in Tier 1 features (within 0 and gap1)
        try:
            # 'last' = last positive effect size feature or second to last feature for top n effect sizes
            last = np.where(rs[order][:-2]<0.)[0][0]
            if last == 0:
                last = n-2
        except IndexError:
            last = n-2

        gap1 = diffs[:last].argmin()+1             # defining most expressed features
        gap2 = diffs[gap1:last+1].argmin()+gap1+1  # defining possibly expressed features


        # creating most expressed feature subtitle label
        tiers_label = "\nTier 1:"
        for _ in np.arange(gap1):
            tiers_label += " | "+self.find_names(self.db.columns[order][_])[0]+" "+str(rs[order][_])[:4]
        tiers_label += "\nTier 2:"
        for _ in np.arange(gap1, gap2):
            tiers_label += " | "+self.find_names(self.db.columns[order][_])[0]+" "+str(rs[order][_])[:4]

        # making effect size plot (a.k.a. most expressed features)
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.barh(y=np.arange(n)[::-1],
                width=rs[order],
                color=np.where(rs_p[order] < self.thrs_p, 'b', 'r'),
                tick_label=self.find_names(self.db.columns[order]))
        # plot some suggested effect thresholds
        ax.vlines(0.1,  -1, n, colors='k', linestyles='-', label='1st effect threshold <= 0.1 (None~small)')
        ax.vlines(0.25, -1, n, colors='k', linestyles='--', label='2nd effect threshold <= 0.25 (small~medium)')
        ax.vlines(0.4,  -1, n, colors='k', linestyles='-.', label='3rd effect threshold <= 0.4 (medium~big)')

        # plot Tier 1 and Tier 2 gaps positions
        ax.plot([-.01, max(0.4, max(rs))], [np.arange(n)[::-1][gap1]+.5, np.arange(n)[::-1][gap1]+.5],
                'm--', label='End of tier 1 features')
        ax.plot([-.01, max(0.4, max(rs))], [np.arange(n)[::-1][gap2]+.5, np.arange(n)[::-1][gap2]+.5],
                'y--', label='End of tier 2 features')

        ax.grid()
        ax.legend(loc='lower right')

        # manually adjust legend
        handles, labels = ax.legend().axes.get_legend_handles_labels()

        handles.append(Patch(facecolor='b', edgecolor='b'))
        labels.append("significant p value (<"+str(self.thrs_p)+")")
        handles.append(Patch(facecolor='r', edgecolor='r'))
        labels.append("NOT significant p value (>"+str(self.thrs_p)+")")
        ax.legend(handles, labels, loc='lower right')


        fig.suptitle("cluster "+str(cluster_number)+" composition")
        fig.tight_layout()

        if self.save_plots:
            fig.savefig(path, facecolor='w')
            
        # reset plot params to default
        plt.rcParams.update({'font.size': old_fontsize})
    

        return tiers_label


    # PLOTS ----------------------------------------------------------------------------------------------


    def plot_features_on_embedding(self, basic_path="./features/"):

        """Summary plot for all features.

        \nbasic_path (str): path at which resulting plots should be stored, if saved

        """
        
        old_fontsize = plt.rcParams['font.size']
        new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
        plt.rcParams.update({'font.size': new_fontsize})


        # plot feature 'd' on the embedding and the real space, showing both its 
        # unprocessed and its processed distributions
        for d in self.db.columns:
            fig, ax = plt.subplots(2, 2, figsize=(20, 10))

            # embedding space
            a = ax[0][0].scatter(*self.embedding.T, c=self.db.loc[:, d], s=self.dot_size, cmap='gist_ncar')
            ax[0][0].grid()
            ax[0][0].set_title('UMAP embedding')

            # real space
            ax[0][1].scatter(self.pos.iloc[:, 0], self.pos.iloc[:, 1], c=self.db.loc[:, d], s=self.dot_size, cmap='gist_ncar')
            ax[0][1].grid()
            ax[0][1].set_title('real position')

            # processed distribution
            ax[1][0].hist(self.db.loc[:, d], bins=100, density=True)
            ax[1][0].grid()
            ax[1][0].set_title('distribution in the preprocessed log space')

            # original distribution
            ax[1][1].hist(self.original_db.loc[self.db.index, d], bins=100, density=True)
            ax[1][1].grid()
            ax[1][1].set_title('original raw distribution')

            # colorbar and final adjustments
            fig.colorbar(a, ax=ax[0][0]).set_label('log of feature expression', rotation=270, labelpad=15)
            fig.suptitle('feature: '+d)
            fig.tight_layout()

            if self.save_plots:
                fig.savefig(basic_path+d+".png", facecolor='w')
                
        # reset plot params to default
        plt.rcParams.update({'font.size': old_fontsize})
    



    # ---------------------------------------------------------------------------------------------------


    def plot_embedding_clusters(self, clusterer, path="./__clusters.png"):

        """Plot HDBSCAN clusters onto real space and UMAP embedding coordinates with appropriate colormap

        \nclusterer (fitted hdbscan.HDBSCAN clustering object): fitted HDBSCAN clustering object used to cluster data from the embedding space
        \npath (str): path where to store the resulting plots if saved

        """

        old_fontsize = plt.rcParams['font.size']
        new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
        plt.rcParams.update({'font.size': new_fontsize})


        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        # scatterplot each single cluster with the respective label
        for cl in self.unique_clusters:
            mask = self.clusters == cl
            ax[0].scatter(*self.embedding[mask].T, s=self.dot_size, color=self.custom_colormap.colors[(1+cl)%len(self.custom_colormap.colors)], label=str(cl))
            ax[1].scatter(self.pos.iloc[:, 0][mask], self.pos.iloc[:, 1][mask], s=self.dot_size, color=self.custom_colormap.colors[(1+cl)%len(self.custom_colormap.colors)])

        # plot title
        fig.suptitle("\nHDBSCAN clustering (eps="+str(clusterer.cluster_selection_epsilon)\
                     +", "+str(100*clusterer.min_samples/len(self.clusters))[:6]+" % of cells for min_clusters)\nclusters: "+str(np.max(self.clusters)+1)+\
                     "   |   noise: "+str(np.where(self.clusters == -1, 1, 0).sum()))

        # show legend iff there are less than 150 total clusters, otherwise it would be unreadable
        if len(np.unique(self.clusters)) < 150:
            lgnd = ax[0].legend(loc='center', bbox_to_anchor=(1.15, 0.5),
                                ncol=4, fancybox=True, shadow=True,
                                title='clusters',
                                fontsize=11)

            # change the feature size manually for the whole legend
            for lh in lgnd.legendHandles:
                lh.set_sizes([200])

        fig.tight_layout()
        if self.save_plots:
            fig.savefig(path, facecolor='w')
            
        # reset plot params to default
        plt.rcParams.update({'font.size': old_fontsize})
    


    # ------------------------------------------------------------------------


    def whole_dataset_summary_plots(self, alpha=0.9, legend_size=200,
                                    plot_with_legend=True, plot_without_legend=True,
                                    plot_noise=True):

        """Plot a Maximum of 4 plots (2 with legend and 2 without legend) which summarize the main features and their self.interpretative_column for each cluster

        \nalpha (float between 0 and 1): transparency for the plots
        \nlegend_size (positive float): size for the dots in the legend
        \nplot_with_legend (bool): whether or not to make the 2 plots with legends (may be unreadable if too many clusters/too long labels are used)
        \nplot_without_legend (bool): whether or not to make the 2 plots without legends (useful if with legends the plot are unreadable, please notice that this parameter is not mandatory to be opposite to 'plot_with_legend')
        \nplot_noise (bool): whether or not to plot the noise cluster if found by HDBSCAN

        """
        
        old_fontsize = plt.rcParams['font.size']
        new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
        plt.rcParams.update({'font.size': new_fontsize})


        # SUMMARY PLOTS FOR THE ADDED COLUMNS, WITH/WITHOUT LEGEND 
        for ii, show in enumerate(['MAIN_FEATURES', 'MAIN_FEATURES', 
                                   self.interpretative_column.upper(), self.interpretative_column.upper()]):
            if not plot_with_legend:
                if ii%2:
                    continue

            if not plot_without_legend:
                if ii%2:
                    continue

            self.verboseprint("Elaborating", show, "plot")

            # prepare color parameters
            if show == 'MAIN_FEATURES':
                color = self.db.loc[:, "MainFeatures"]
            else:
                color = self.db.loc[:, self.interpretative_column]
            n_clust = len(np.unique(color))


            fig, ax = plt.subplots(1, 1, figsize=(19,9))

            # actual plot
            for i, column in enumerate(np.unique(color)):
                if column == '':
                    if plot_noise:
                        ax.scatter(self.pos.iloc[:, 0][color==column], self.pos.iloc[:, 1][color==column],
                                   alpha=alpha, color='black',
                                   label=' Noise', s=self.dot_size)
                    else:
                        continue
                ax.scatter(self.pos.iloc[:, 0][color==column], self.pos.iloc[:, 1][color==column],
                           alpha=alpha, color=self.custom_colormap.colors[i%len(self.custom_colormap.colors)],
                           label=np.unique(color)[i], s=self.dot_size)

            # plot legend iff ii is odd
            if ii%2:
                lgnd = ax.legend(loc='center left', bbox_to_anchor=(1., 0.5),
                                 ncol=1+n_clust//31, fancybox=True, shadow=True,
                                 title=show,
                                 fontsize=11)

                #change the feature size manually for the whole legend
                for lh in lgnd.legendHandles:
                    lh.set_sizes([legend_size])


            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Real space")
            ax.grid()


            fig.suptitle("Cell type classification - Area "+str(self.area)+" - "+show.upper(),
                         fontsize=20)

            fig.tight_layout()

            if self.save_plots:        
                if ii%2:
                    fig.savefig(self.folder+show.upper()+".png", facecolor='w')
                else:
                    fig.savefig(self.folder+show.upper()+"_nolegend.png", facecolor='w')

        # reset plot params to default
        plt.rcParams.update({'font.size': old_fontsize})
    

    # ------------------------------------------------------------------------

    
    def find_monitoring_features(self):
        
        """Find which features are the main ones accoring to self.importance_column"""
        
        # Find and order important features (in the immunofluorescence specific case it could be lineage defining features)
        # which will be used to monitor the nature of each cluster and compare it to the whole sample
        monitoring_features = []

        for column in self.db.columns:
            try:
                feature = [a for a in self.reference.loc[:, self.correspondence_column] if column in a][0]
                if self.reference[self.reference.loc[:, self.correspondence_column] == feature ].loc[:, self.importance_column].iloc[0]:
                    monitoring_features.append(column)
            except IndexError:
                self.verboseprint(column, "not present in reference file. Cannot establish if it is a monitoring feature")
                continue


        # order features alphabetically to always have the same set across different samples
        self.monitoring_features = np.asarray(monitoring_features)[np.argsort(monitoring_features)]

        
    

    def plot_cluster_spatial_location(self, tiers_label, mask, cluster_number=-2,
                                      path="./cluster__position.png"):

        """Plot cluster position in UMAP embedding and in real space, together with a summary of Tier1 and Tier2 features.

        \ntiers_label (str): string produced by func containing Tier 1 and Tier 2 features, which will be used as header for the plot 
        \nmask (array-like of boolean values): boolean values corresponding to whether or not a single cell is part of the considered cluster
        \ncluster_number (integer, non-negative): number to use as header for cluster name, useful to distinguish multiple clusters.
        \npath (str): where to store the resulting plot (if saved)

        """

        old_fontsize = plt.rcParams['font.size']
        new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
        plt.rcParams.update({'font.size': new_fontsize})

        
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        # baseline light grey plot for whole sample
        ax[0].scatter(self.embedding.T[0], self.embedding.T[1], s=self.dot_size/2, 
                      color='lightgrey', label='other')
        ax[1].scatter(self.pos.iloc[:, 0], self.pos.iloc[:, 1], s=self.dot_size/2, 
                      color='lightgrey')

        # cluster plot with vivid red and bigger dots
        ax[0].scatter(self.embedding.T[0][mask], self.embedding.T[1][mask], s=self.dot_size*2, color='r', label=str(cluster_number))
        ax[1].scatter(self.pos.iloc[:, 0][mask], self.pos.iloc[:, 1][mask], s=self.dot_size*2, color='r')

        ax[0].set_title("UMAP embedding")
        ax[1].set_title("Real position")


        fig.suptitle("Cluster "+str(cluster_number)+": containing "+str(np.asarray(mask).sum())+\
                     " cells. Reporting feature name and effect size"+tiers_label)

        lgnd = ax[0].legend(loc='center', bbox_to_anchor=(1., 0.5),
                            ncol=1, fancybox=True, shadow=True,
                            title='clusters',
                            fontsize=11)

        #change the feature size manually for the whole legend
        for lh in lgnd.legendHandles:
            lh.set_sizes([200])

        fig.tight_layout()

        if self.save_plots:
            fig.savefig(path, facecolor='w')
    
        # reset plot params to default
        plt.rcParams.update({'font.size': old_fontsize})
    

    # -------------------------------------------------------------------------


    def monitoring_features_plot(self, mask, cluster_number=-2, 
                                 palette=[(0.2, 0.5, 1.), (1., 0.2, 0.2)], 
                                 path='./cluster__monitoring_featuresining.png'):

        """Plot Kernel Density Estimation to compare a cluster with whole dataset using some monitoring features.

        \nmask (array-like of boolean values): boolean values corresponding to whether or not a single cell is part of the considered cluster
        \ncluster_number (integer, non-negative): number to use as header for cluster name, useful to distinguish multiple clusters.
        \nmonitoring_features (list or array-like): list of features over which each cluster will be compared with whole sample using KDE plots
        \npalette (list of 2 rgb tuples): color palette to use for cluster or whole sample cells and KDE plots
        \npath (str): where to store the resulting plot (if saved)

        """    
        
        old_fontsize = plt.rcParams['font.size']
        new_fontsize = 15  # font size to be used in all plots where not specified, we suggest to use larger values than default as most plots contain multiple labels and texts that are helpful for the results interpretation.
        plt.rcParams.update({'font.size': new_fontsize})


        # build 'sub_db' for monitoring features plot, such pandas DataFrame will be composed
        # of only the monitoring features, and will contain an extra column based on which 
        # cells belong to a cluster or to the whole sample (cluster cells will have duplicates)
        sub_db = pd.DataFrame({'expression': self.original_db.loc[:, 
                               self.monitoring_features].T.stack().values, 
                               'feature': [a[0] for a in self.original_db.loc[:,
                               self.monitoring_features].T.stack().index],
                               'cluster': list(np.where(mask, 
                                                        "cluster "+str(cluster_number),
                                                        'whole sample')) * len(self.monitoring_features)})
        sub_db2 = sub_db[sub_db.loc[:,'cluster']!='whole sample']
        sub_db2.loc[:,'cluster'] = 'whole sample'
        sub_db = pd.concat([sub_db, sub_db2])

        plt.figure(figsize=(40, 45))

        # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
        g = sns.FacetGrid(data=sub_db, row='feature', hue='cluster', palette=palette,
                          sharey=False, legend_out=True, height=1.01, 
                          aspect=2*len(self.monitoring_features)-2)

        # then we add the densities kdeplots for each month
        g.map(sns.kdeplot, 'expression',
              bw_adjust=1, clip_on=False,
              fill=True, alpha=0.4, linewidth=2)

        # here we add a horizontal line for each plot
        g.map(plt.axhline, y=0, lw=5, clip_on=False)


        ft = 25 # bigger font size for some within plot headers

        # we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
        # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
        for ii, axis in enumerate(g.axes.flat):


            axis.text(-self.original_db.quantile(0.05).max(), 0.0, self.monitoring_features[ii],
                      fontweight='bold', fontsize=ft, color=axis.lines[0].get_color())
            axis.text(self.original_db.quantile(0.8).max(), 0.0, 'effect size: '+str(self.res.loc[cluster_number, self.monitoring_features[ii]])[:6],
                      fontweight='bold', fontsize=ft, color=axis.lines[0].get_color())
            axis.plot([sub_db2.iloc[:, :-1].groupby('feature').median().loc[self.monitoring_features[ii]],
                       sub_db2.iloc[:, :-1].groupby('feature').median().loc[self.monitoring_features[ii]]],
                       axis.get_ylim(), 'r--', linewidth=3)
            axis.plot([sub_db.iloc[:, :-1].groupby('feature').median().loc[self.monitoring_features[ii]],
                       sub_db.iloc[:, :-1].groupby('feature').median().loc[self.monitoring_features[ii]]],
                      axis.get_ylim(), 'b--', linewidth=3)
            axis.set_ylabel("")

        # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
        g.fig.subplots_adjust(hspace=-0.)

        # eventually we remove axes titles, yticks and spines
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)

        plt.setp(axis.get_xticklabels(), fontsize=ft, fontweight='bold')
        plt.xlabel('feature Expression (a.u.)', fontweight='bold', fontsize=ft)
        g.fig.suptitle("", ha='right', fontsize=ft+5, fontweight=15)
        plt.xlim(-self.original_db.quantile(0.05).max(),
                 self.original_db.quantile(0.95).max())
        plt.xticks(fontweight='bold', fontsize=ft//2)
        plt.legend()

        # manually adjust the legend labels
        handles, labels = [], []
        handles.append(Patch(facecolor='b', edgecolor='b', alpha=0.4))
        labels.append("whole sample")    
        handles.append(Patch(facecolor='r', edgecolor='r', alpha=0.4))
        labels.append("cluster "+str(cluster_number))
        handles.append(Line2D([0], [0], linestyle='--', linewidth=5, color='b'))
        labels.append("feature median (whole sample)")
        handles.append(Line2D([0], [0], linestyle='--', linewidth=5, color='r'))
        labels.append("feature median (cluster)")

        plt.legend(handles, labels, ncol=2, loc='upper center',
                   bbox_to_anchor=(0.75, len(self.monitoring_features)+2.01), fontsize=ft)


        if self.save_plots:
            fig = plt.gcf()
            fig.savefig(path, facecolor='w', bbox_inches='tight')       

        
        # reset plot params to default
        plt.rcParams.update({'font.size': old_fontsize})
        
    



    def run_pipeline(self):
        """Run the complete BRAQUE pipeline from preprocessing to plots and storing results."""

        # =============================================================================
        # PREPROCESSING
        # =============================================================================

        if not self.load_db:
            # discard unwanted features, due to different possible reasons
            if self.perform_features_selection:
                print("Features Selection ...", datetime.now())
                self.features_selection()

                # measure the number of the remaining features

            # # perform Lognormal Shrinkage (LNS) from Dall'Olio et al. (2023) :https://doi.org/10.3390/e25020354
            if self.perform_lognormal_shrinkage:
                print("Lognormal Shrinkage ...", datetime.now())
                self.lognormal_shrinkage(subsampling=self.subsampling, 
                                         max_n_gaussians=self.max_n_gaussians,
                                         contraction_factor=self.contraction_factor,
                                         populations_plot=self.populations_plot)

            else:
                self.verboseprint("Not performing data transformation.")



        # =============================================================================
        # UMAP EMBEDDING
        # =============================================================================


        if not self.load_embed:
            print("UMAP embedding ...", datetime.now())
            self.embed_dataset(save_embed=True, nn=self.nn, metric=self.metric)


        self.plot_features_on_embedding(basic_path=self.folder+"features/")





        # =============================================================================
        # HDBSCAN clustering on UMAP embedding
        # =============================================================================

        if not self.load_clusters:
            print("HDBSCAN clustering ...", datetime.now())
           
            mindim = max(int(len(self.db)*0.00005), 10)  # minimum number of cells allowed to form a speparate cluster [0.05%,0.2%]

            clusterer = hdbscan.HDBSCAN(min_cluster_size=mindim, min_samples=mindim,
                                        cluster_selection_epsilon=self.HDBSCAN_merging_parameter,  # between 0.08 and 0.12 usually
                                        cluster_selection_method='eom')
            self.clusters = clusterer.fit_predict(self.embedding)
            self.unique_clusters = np.unique(self.clusters)
            self.clusterer = clusterer

        self.plot_embedding_clusters(clusterer, path=self.folder+self.area+"__clusters.png")

        self.verboseprint("End of clustering...", datetime.now())



        # optional reclustering procedure <Specific for the paper>
        # tune another clustering with indipendent parameters on the biggest cluster
        # (which probably needs to be furtherly splitted)
        if self.reclustering_step:
            if not self.load_clusters:
                print("Reclustering step ...", datetime.now())



                # Find biggest cluster as candidate for the reclustering procedure
                sizes = self.db.groupby(self.clusters).size()
                if sizes.max() > len(self.db)*0.1:
                    sub_db = self.db[self.clusters == sizes.argmax()-1].copy()
                    sub_pos = self.pos[self.clusters == sizes.argmax()-1].copy()
                    candidate_reclustering_core = sizes.argmax()-1

                else:
                    candidate_reclustering_core = -1


                # evaluate if biggest cluster is candidate for further splitting,
                # and in that case perform a reclustering of just the biggest cluster
                if candidate_reclustering_core != -1 and sizes.max() > len(self.db)/10:
                    mask = self.clusters == candidate_reclustering_core
                    sub_db = self.db[mask].copy()
                    sub_embedding = self.embedding[mask].copy()
                    sub_pos = self.pos[mask].copy()

                    sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=mindim*10, min_samples=mindim*10,
                                                    cluster_selection_epsilon=self.HDBSCAN_merging_parameter*0.8,  # 0.08 for whole Lymph node
                                                    cluster_selection_method='leaf')
                    sub_clusters = sub_clusterer.fit_predict(sub_embedding)

        #                 plot_embedding_clusters(sub_clusters, sub_pos, sub_embedding, sub_clusterer, 
        #                                         path=folder+self.area+"__clusters_reclustering_core_zoom.png")



            # if optional reclustering procedure has been performed, 
            # overwrite biggest cluster labels with new ones <Specific for the paper>

            if not self.load_clusters:

                if candidate_reclustering_core != -1 and sizes.max() > len(self.db)/10:

                    sub_clusters = np.where(sub_clusters < 1, sub_clusters, 
                                            sub_clusters+max(self.clusters))
                    sub_clusters = np.where(sub_clusters == 0, candidate_reclustering_core, 
                                            sub_clusters)

                    self.clusters[mask] = sub_clusters
                    self.unique_clusters = np.unique(self.clusters)


                    self.plot_embedding_clusters(clusterer,
                                                 path=self.folder+self.area+"__clusters.png")

                    self.verboseprint("Computing features "+self.interpretative_column+" and effect size", 
                                      datetime.now())

                    
                    


        # =============================================================================
        # ASSIGN CLUSTER interpretative_column
        # =============================================================================

        # thresholds for pvalues (statistical interpretative_column) and for effect sizes (difference magnitude)
        # using Bonferroni correction, most stringent one w.r.t. false positivies (e.g. falsely expressed features)
        self.thrs_p = self.p_val_basic_threshold/len(self.db.columns)/len(self.unique_clusters)
        
        print("Computing features importance per cluster...", datetime.now())
        self.features_importance_per_cluster(select='original', compare_with='rest')
        self.find_monitoring_features()




        
        
        
        # =============================================================================
        # PLOTS SECTION
        # =============================================================================

        print("Plots section ...", datetime.now())

        self.verboseprint("clusters:", np.unique(self.clusters))

        self.verboseprint("Producing results for whole sample plots")
        self.add_main_features_interpretative_columns()
        self.whole_dataset_summary_plots()

        for i in np.unique(self.clusters):
            self.verboseprint("producing results for cluster", i)

            # select cells within cluster i-th
            mask = self.clusters == i  

            # select res and res_p corresponding to cluster i-th
            rs = np.asarray(self.res.loc[i, :])
            rs_p = np.asarray(self.res_p.loc[i, :])

            n = np.min([30, len(rs)])  # consider a fixed maximum number of features

            # making effect sizes plot (ranking features in descending order of effect size)
            # and building the 'tiers_label' variable, 
            # a string with Tier 1 (probably important features)
            # and Tier 2 (possibly important features)
            tiers_label = self.most_important_cluster_effect_sizes(rs=rs, rs_p=rs_p, n=n,
                                                                   cluster_number=i,
                                                                   path=self.folder+"expressions/"+self.area+"__cluster_"+str(i)+"_expression.png")


            # making postion plot (where are cluster cells) 
            # and reporting most expressed features
            self.plot_cluster_spatial_location(tiers_label=tiers_label, 
                                               mask=mask, cluster_number=i, path=self.folder+"positions/"+self.area+"__cluster_"+str(i)+"_position.png")


            # Monitoring features plot
            self.monitoring_features_plot(mask=mask, cluster_number=i, path=self.folder+"expressions/"+self.area+"__cluster_"+str(i)+"_monitoring_featuresining.png")


        self.verboseprint("Saving outcomes", datetime.now())





        # =============================================================================
        # SAVE OUTCOMES
        # =============================================================================

        # save results
        if not self.load_clusters:
            print("Saving results ...", datetime.now())
            # cluster labels
            pd.DataFrame(self.clusters, columns=['cluster']).to_csv(self.folder+self.area+"__clusters_labels.csv")

            # cluster sizes
            sizes = self.db.groupby(self.clusters).size()
            sizes.to_csv(self.folder+self.area+"__clusters_sizes.csv", index_label='cluster')
            self.verboseprint("End of pipeline", datetime.now())
            
            
            
        
        print("Pipeline completed!", datetime.now())
