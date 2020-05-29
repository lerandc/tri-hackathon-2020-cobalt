from camd.agent.base import HypothesisAgent
    
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
        candidate_data = candidate_data.sort_values('bandgap_pred', ascending=True)
        top_candidates = candidate_data.head(number)
        return top_candidates


def select_random_candidates(candidate_data, number):
        random_candidates = candidate_data.sample(n=number)
        return random_candidates


class BandgapAgent(HypothesisAgent):
    
    def __init__(self, regressor, num=5, random=False):
        self.regressor = regressor
        self.random = random
        self.num = num
        
    def get_hypotheses(self, candidate_data, seed_data):
        # order of candidate and seed matters!

        # Fit on known data
        x_known = get_features_from_df(seed_data)
        y_known = seed_data['bandgap']
        self.regressor.fit(x_known, y_known)
        
        # Predict unknown data
        x_unknown = get_features_from_df(candidate_data)
        y_predicted = self.regressor.predict(x_unknown)
        
        # Pick top 5 candidates
        # Potential changes: predict areas far from explored ones,...
        candidate_data['bandgap_pred'] = y_predicted
        candidate_data = candidate_data[candidate_data['bandgap_pred'] > 0.0]
    
        if self.random:
            return select_random_candidates(candidate_data, self.num)
        elif not self.random: 
            return select_candidates(candidate_data, self.num)
    

    

        
        