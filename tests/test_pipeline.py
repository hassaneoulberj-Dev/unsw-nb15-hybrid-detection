import unittest
import numpy as np
import pandas as pd
from src.training import feature_frame, group_split, make_preprocessor, hidden_features
from sklearn.neural_network import MLPClassifier


class PipelineTests(unittest.TestCase):
    def test_target_columns_never_enter_features(self):
        df = pd.DataFrame({"id":[1,2], "label":[0,1], "attack_cat":["Normal","Attack"], "dur":[1.,2.]})
        self.assertEqual(list(feature_frame(df)), ["dur"])

    def test_unseen_categories_and_test_values_do_not_change_fit(self):
        x = pd.DataFrame({"dur":[1.,2.,3.,4.], "proto":["tcp","udp","tcp","udp"]})
        prep = make_preprocessor(x, 2).fit(x, [0,1,0,1])
        scaler = prep.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
        before = scaler.mean_.copy()
        values = prep.transform(pd.DataFrame({"dur":[1000000.], "proto":["never_seen"]}))
        np.testing.assert_array_equal(before, scaler.mean_)
        self.assertTrue(np.isfinite(values).all())

    def test_duplicate_records_cannot_cross_validation_boundary(self):
        df = pd.DataFrame({"id":range(200), "dur":np.tile(np.arange(100),2), "label":np.tile(np.arange(100)%2,2)})
        fit, val = group_split(df)
        self.assertFalse(set(df.iloc[fit].dur) & set(df.iloc[val].dur))

    def test_hidden_representation_reproduces_network_probability(self):
        rng = np.random.default_rng(42); x=rng.normal(size=(80,5)); y=(x[:,0]>0).astype(int)
        model=MLPClassifier(hidden_layer_sizes=(8,4),random_state=42)
        model.partial_fit(x,y,classes=[0,1])
        h=hidden_features(model,x)
        logits=h @ model.coefs_[-1]+model.intercepts_[-1]
        probabilities=1/(1+np.exp(-logits))
        np.testing.assert_allclose(probabilities[:,0],model.predict_proba(x)[:,1],rtol=1e-6)


if __name__ == "__main__": unittest.main()
