import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import make_scorer, accuracy_score, DistanceMetric
from sklearn.metrics._scorer import _SCORERS

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from datetime import datetime


class BaseModel():

    def __init__(self, params={}, data_name=None):
        self.curr_best_score = None
        self.params = params
        self.data_name = data_name
        self.model = None

    def train_model_dt(self, X_train, y_train):
        print('Training Decision Trees... ')
        self.curr_best_score = 0
        model = DecisionTreeClassifier()
        scaler = StandardScaler()
        pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])
        g_params = {'model__criterion': ['gini', 'entropy'],
                    'model__ccp_alpha': np.arange(0, 0.6, 0.05),
                    'model__max_depth': list(range(5, 50, 5))}
        self.train_model_curves(X_train, y_train, pipe, g_params, title='Decision Tree')
        model.predict()

    def train_model_nn(self, X_train, y_train):
        print('Training Neural networks... ')
        self.curr_best_score = 0
        model = MLPClassifier(hidden_layer_sizes=100, alpha=0.0001, learning_rate_init=0.1, random_state=1,
                              max_iter=300)
        scaler = StandardScaler()
        pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])
        g_params = {'model__hidden_layer_sizes': np.arange(100, 1000, 100),
                    'model__learning_rate_init': list(np.arange(0.1, 0.6, 0.1))}
        self.train_model_curves(X_train, y_train, pipe, g_params, title='Neural networks MLP')

    def train_model_svm(self, X_train, y_train):
        print('Training support vector machines... ')
        self.curr_best_score = 0
        model = SVC()
        scaler = StandardScaler()
        pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])
        g_params = {'model__C': np.arange(0.1, 1., 0.1),
                    'model__kernel': ['linear', 'poly', 'rbf']}
        self.train_model_curves(X_train, y_train, pipe, g_params, title='SVM SVC')

    def train_model_knn(self, X_train, y_train):
        print('Training K nearest neighbors... ')
        model = KNeighborsClassifier(n_neighbors=10)
        scaler = StandardScaler()
        pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])
        g_params = [{'model__n_neighbors': np.arange(5, 25, 5),
                     'model__weights': ['uniform', 'distance'],
                     'model__metric': ['euclidean', 'chebyshev']

                     },
                    {'model__n_neighbors': np.arange(5, 25, 5),
                     'model__weights': ['uniform', 'distance'],
                     'model__metric': ['seuclidean'],
                     'model__metric_params': [{'V': np.array(np.var(X_train, axis=0, ddof=1))}]
                     }]
        self.curr_best_score = 0

        for g_param in g_params:
            best_estimator, best_score = self.train_model_curves(X_train, y_train, pipe, g_param,
                                                                 title='K nearest neighbours')
        print('Current best estimator ', self.curr_best_score, self.model)
        """
        model_params = {}
        for m_p in best_params.keys():
            model_params[m_p.replace('model__', '')] = best_params[m_p]
        best_model = KNeighborsClassifier(model_params)
        """

    def train_model_boost(self, X_train, y_train):
        self.curr_best_score = 0
        print('Training ensemble learning of Decision trees, with adaboost ')
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5))
        scaler = StandardScaler()
        pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])
        g_params = {'model__n_estimators': np.arange(10, 100, 10),
                    'model__learning_rate': np.arange(0.1, 1., 0.1)}
        self.train_model_curves(X_train, y_train, pipe, g_params, title='Boosting Adaboost')

    def train_model_curves(self, X_train, y_train, pipe, params, title, cv=8):
        title = title + ' For ' + self.data_name
        cv = cv
        g_params = params
        # print(sorted(SCORERS.keys()))
        scoring = {"Accuracy": make_scorer(accuracy_score)}
        gscv = GridSearchCV(pipe, param_grid=g_params, cv=cv, scoring='accuracy', return_train_score=True, refit=True)
        gscv.fit(X_train, y_train)

        # best_params = gscv.best_params_
        train_sizes, train_score, val_score, fit_times, score_times = learning_curve(gscv.best_estimator_, X_train,
                                                                                     y_train, return_times=True)
        # print('Train sizes, train score, val score', train_sizes, train_score, val_score)
        self.simple_plot(train_score, val_score, title, category='Score')
        self.simple_plot(fit_times, score_times, title, category='Time')
        # self.simple_plot(fit_times)
        # clf.predict()
        # gscv.fit(X_train, y_train)
        # gscv.error_score

        df = pd.DataFrame(gscv.cv_results_)
        print(gscv.cv_results_)
        with open("dt_results.txt", "a") as dtcsv:
            dtcsv.write(df.to_csv())

        with open("scores.txt", "a") as myfile:
            myfile.write('\n' + title + ' Best params ' + str(gscv.best_params_) + str(gscv.best_score_))

        print('Best params ', gscv.best_params_, gscv.best_score_)
        # print('Data frame', df.keys())
        """
        
        print(df.keys())
        print('No of splits per iteration ', gscv.n_splits_)
        
        print('best esti',  gscv.best_estimator_)
        """
        if gscv.best_score_ > self.curr_best_score:
            self.curr_best_score = gscv.best_score_
            self.model = gscv.best_estimator_
        self.plotting(g_params, df, title)
        return gscv.best_estimator_, gscv.best_score_

    def plotting(self, grid_params, df, title):
        """
        Referred from below
        https://stackoverflow.com/questions/62363657/how-can-i-plot-validation-curves-using-the-results-from-gridsearchcv#
            :~:text=Validation%20Curve%20is%20meant%20to,the%20impact%20of%20each%20parameter.

        :param grid_params:
        :param df:
        :return:
        """

        results = ['mean_test_score',
                   'mean_train_score',
                   'std_test_score',
                   'std_train_score']

        def pooled_var(stds):
            # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
            n = 5  # size of each group
            return np.sqrt(sum((n - 1) * (stds ** 2)) / len(stds) * (n - 1))

        fig, axes = plt.subplots(1, len(grid_params),
                                 figsize=(5 * len(grid_params), 7),
                                 sharey='row')
        axes[0].set_ylabel("Score", fontsize=20)

        for idx, (param_name, param_range) in enumerate(grid_params.items()):

            if len(param_range) > 0 and (type(param_range[0]) is dict):
                continue;
            if isinstance(param_range, dict) or (type(param_range) is dict):
                continue
            grouped_df = df.groupby(f'param_{param_name}')[results] \
                .agg({'mean_train_score': 'mean',
                      'mean_test_score': 'mean',
                      'std_train_score': pooled_var,
                      'std_test_score': pooled_var})

            previous_group = df.groupby(f'param_{param_name}')[results]
            axes[idx].set_xlabel(param_name, fontsize=15)
            axes[idx].set_ylim(0.0, 1.1)
            lw = 2
            axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                           color="darkorange", lw=lw)
            axes[idx].fill_between(param_range, grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                                   grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
                                   color="darkorange", lw=lw)
            axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                           color="navy", lw=lw)
            axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                                   grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
                                   color="navy", lw=lw)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle('Validation curves ' + title, fontsize=18)
        # fig.legend(handles, labels, loc='best', ncol=2, fontsize=20)
        fig.legend()

        fig.subplots_adjust(bottom=0.25, top=0.85)
        # plt.show()
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig('plots/validation/Validation_' + title + '_' + date_time + '.png')

    def simple_plot(self, train_score, val_score, title, category, n_ticks=0, n_cvs=0):
        val_score = np.array(val_score).T
        train_score = np.array(train_score).T

        fig, ax = plt.subplots(1, len(train_score), figsize=(5 * len(train_score), 7), sharex=True, sharey=True)

        ax[0].set_ylabel(category, fontsize=20)
        for ix in range(len(train_score)):
            ax[ix].set_xlabel('CV ' + str(ix), fontsize=10)
            # ax[ix].set_ylim(0.0, 1.1)
            ax[ix].plot(train_score[ix], label=str('Training '))
            ax[ix].plot(val_score[ix], label=str('Validate '))
            # ax[ix].plot(fit_times[ix], label=str('Fit times ' ))
            # ax[ix].legend()
            """
            plt.plot(train_score[ix], label=str('Train score for cv'+ str(ix)))
            plt.plot(val_score[ix], label=str('Val score ' + str(ix)))
            plt.legend()
            plt.title(title)
            plt.show()
            """
        fig.suptitle('Learning curve ' + category + ' ' + title, fontsize=20)
        fig.legend()
        fig.subplots_adjust(bottom=0.25, top=0.85)
        # plt.show()
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig('plots/learning/Learning_' + title + '_' + category + '_' + date_time + '.png')

    def plot_scores(self, train_score, val_score, cv_val=None, p_range=None):
        # print('Train score', train_score)
        print('Val score', val_score)
        print(' arg max', np.unravel_index(np.argmax(val_score), np.array(val_score).shape))
        if cv_val > len(p_range):
            for ix in range(len(p_range)):
                plt.plot(train_score[ix], label=str('Train for cv' + str(p_range[ix])))
                plt.plot(val_score[ix], label=str('Val score for cv' + str(p_range[ix])))
        else:
            train_score = np.array(train_score).T
            val_score = np.array(val_score).T
            for ix in range(cv_val):
                plt.plot(train_score[ix], label=str('Train for cv ' + str(ix)))
                plt.plot(val_score[ix], label=str('Val score for cv ' + str(ix)))
            plt.xticks(ticks=range(len(p_range)), labels=p_range)
        plt.legend()
        plt.show()

    def test_model(self, X_test, y_test):
        model_score = self.model.score(X_test, y_test)
        with open("scores.txt", "a") as myfile:
            myfile.write('\n' + ' Test score of model ' + str(model_score))
        print('Test Score.. ', model_score)
