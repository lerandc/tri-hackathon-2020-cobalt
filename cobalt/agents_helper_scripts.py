from camd.agent.base import HypothesisAgent
import pandas as pd
import numpy as np

def get_features_from_df(df):
    """
    Function that extract only feature columns of the dataframe
    
    Args:
        df           Pandad dataframe of the dataset, including samples, 
                     their features and labels.
        
    Returns:
        features_df  df contains only features of the samples. 
    
    """
    features_df = df.drop(['vae','bandgap','spectrum','bandgap_pred'], axis=1, errors='ignore')
    return features_df


def select_candidates(candidate_data, number):
    """
    Select top candidates from candidate dataframe, based off of lowest bandgap.
    """
    candidate_data = candidate_data.sort_values('bandgap_pred', ascending=True)
    top_candidates = candidate_data.head(number)
    return top_candidates


def select_random_candidates(candidate_data, number):
    """
    Select random candidates from candidate dataframe
    """
    random_candidates = candidate_data.sample(n=number)
    return random_candidates
    
def explore_candidates(candidate_data, seed_data, number):
    """
    Explore dataset by looking for samples that are compositions most dissimilar to the compositions of the existing seed data
    
    Ideal way would be to some sort of clustering in composition space,
    then sample with frequencies corresponding to the size of the subspaces
    
    Algorithm is as follows:
    -Select 10*number random candidates
    -Compute mean vector similarity (cosine similarity, absolute difference) 
    for each 10*num candidates with respect to the rest of the candidate set
    -Sort 10*num candidates by similarity
    -Choose num final candidates with lowest similarity score
    """
    random_candidates = candidate_data.sample(n=10*number)
    all_comps = get_features_from_df(seed_data)
    scores = []
    for i in range(random_candidates.shape[0]):
        candidate = pd.DataFrame(random_candidates.iloc[i])
        target_comp = candidate.drop(['bandgap', 'vae', 'spectrum', 'bandgap_pred'])
        tmp_comps = np.abs(all_comps - target_comp.values.squeeze())
        tmp_comps = tmp_comps.sum(axis=1)
        scores.append(np.mean(tmp_comps))

    #largest absolute difference is greatest dissimilarity
    random_candidates['score'] = scores
    random_candidates = random_candidates.sort_values('score', ascending=False)
    random_candidates = random_candidates.drop('score', axis=1)
    return random_candidates.head(number)


class BandgapAgent(HypothesisAgent):
    """
    Custom agent that can suggest experimental candidates based on mixed exploration/exploitation strategies.
    By default, 10% of experimental suggestions are suggested by exploration, 90% by exploitation
    
    Agent can also return random candidates (for baseline measurement)
    """
    
    def __init__(self, regressor, num=5, random=False, explore=0.1):
        """
        Expects a scikit learn style regressor
        Num is number of total suggested candidates
        Random = True to suggest random samples
        explore is the fraction of suggested candidates that are drawn from exploration strategy
        """
        self.regressor = regressor
        self.random = random
        self.num = num
        self.explore = np.int(np.ceil(num*explore))
        self.exploit = np.int(num - self.explore)
        
    def get_hypotheses(self, candidate_data, seed_data):

        # Fit on known data
        x_known = get_features_from_df(seed_data)
        y_known = seed_data['bandgap']
        self.regressor.fit(x_known, y_known)
        
        # Predict unknown data
        x_unknown = get_features_from_df(candidate_data)
        y_predicted = self.regressor.predict(x_unknown)
        
        # Pick top 5 candidates
        candidate_data['bandgap_pred'] = y_predicted
        candidate_data = candidate_data[candidate_data['bandgap_pred'] > 0.0]
        
        if self.random:
            return select_random_candidates(candidate_data, self.num)
        elif not self.random: 
            selection = select_candidates(candidate_data, self.exploit)
            selection = pd.concat([selection, explore_candidates(candidate_data, self.explore)], axis=0)
            return selection

    

        
        