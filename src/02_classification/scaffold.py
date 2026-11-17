import numpy as np

# This is the version number so you can align your saved weights and expected features.
# You don't have to change this.
FEATURE_VERSION = 2  # 2 = Changes 1-6 (15 features)

##Change 4: constants for the braking features
DRY_MASS = 27500  # kg, not in physics.pdf, estimated from burns in the data
G = 9.80665  # m/s², physics.pdf
BURN_THRUST = 525560  # N, the expert's usual burn
Y_GROUND = 18.66  # m, lowest y in the data = ground
##Change 6: how fast the near-ground boost fades with height
BOOST_HEIGHT = 10  # m, boost is 1 at the ground, 0.37 at 10 m, ~0 above 30 m
##Change 8: thrust of each action as a share of full thrust (0 off, 1-3 low, 4-6 medium, 7-9 full)
THRUST_LEVEL = np.array([0, 0.62, 0.62, 0.62, 0.81, 0.81, 0.81, 1.0, 1.0, 1.0])

# DO NOT CHANGE THE CLASS NAME
# It is required by the online evaluation
class Controller:
    """
    This is the scaffold class for your controller.
    
    It is complete for a linear model with softmax output. You must make at least 2 informed changes to the class:
    - This could be polynomial features
    - It could be lag features (and as such you may need to store a history of observations)
    - It could be RBF features or even custom physics-informed features
    
    Note: Normalisation, regularisation do not count as informed changes.
    """
    # The __init__ method can contain any parameters you want to save or use later.
    # Be aware that the online evaluation will call the default constructor, so any parameters you want to use must have default values.
    # For example, if you find that you need 1000 epochs then you must set this as the default if you intend to train it in the browser.
    ##Change 3: weight rare actions more in the loss (action 2 is ~66 % of the data).
    # 0 = off, 0.5 = square root of balanced, 1 = balanced. Only used in fit.
    # L2: pulls the weights towards 0, bias excluded. 0 = off.
    ##Change 7: extra loss weight for single actions on top of the class weights, e.g. {8: 1.5}.
    # The learner brakes too little, action 8 = full thrust straight.
    ##Change 8: urgency_strength > 0 weights rows by how urgent braking is (fast and close to the ground)
    # times the thrust of the action, so 8 counts more than 5, 5 more than 2, 0 never. 0 = off.
    def __init__(self, learning_rate=0.1, epochs=10, class_weight_power=0.0, l2_lambda=0.0, action_boost=None,
                 urgency_strength=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_weight_power = class_weight_power
        self.l2_lambda = l2_lambda
        self.action_boost = dict(action_boost or {})
        self.urgency_strength = urgency_strength
        self.weights = None
        self.train_loss_history = []
        self.validation_loss_history = []

    def save(self, path):
        """
        Save trained parameters for later evaluation.
        The usage is like this:
            model.save("controller.npz") 
        Then you can upload the file to the online evaluation.
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        with open(path, "wb") as target:
            # Add anything your model needs to save here. 
            # You can add more fields to the np.savez_compressed call if you need to save more parameters.
            np.savez_compressed(
                target, 
                feature_version=FEATURE_VERSION, 
                weights=self.weights, 
                classes=self.classes_,
                learning_rate=self.learning_rate, 
                epochs=self.epochs,
                class_weight_power=self.class_weight_power,
                l2_lambda=self.l2_lambda,
                action_boost_actions=list(self.action_boost.keys()),
                action_boost_factors=list(self.action_boost.values()),
                urgency_strength=self.urgency_strength,
                train_loss_history=self.train_loss_history,
                validation_loss_history=self.validation_loss_history,
                mean=self.X_mean_, 
                std=self.X_std_
            )

    def load(self, path):
        """
        Restore a saved model. The usage is like this:
            model.load("controller.npz")
        """
        with np.load(path, allow_pickle=False) as saved:
            if 'feature_version' not in saved or int(saved['feature_version']) != FEATURE_VERSION:
                raise ValueError('Saved weights use an incompatible feature version; retrain and save the model again.')
            # Again if you add more fields you need to save you must update this to load them as well.
            self.weights = saved["weights"]
            self.classes_ = saved["classes"] if "classes" in saved else np.arange(self.weights.shape[1])
            self.learning_rate = float(saved["learning_rate"])
            self.epochs = int(saved["epochs"])
            # Files saved before Change 3 don't have it
            self.class_weight_power = float(saved["class_weight_power"]) if "class_weight_power" in saved else 0.0
            self.l2_lambda = float(saved["l2_lambda"]) if "l2_lambda" in saved else 0.0
            if "action_boost_actions" in saved:
                self.action_boost = dict(zip(saved["action_boost_actions"].tolist(), saved["action_boost_factors"].tolist()))
            else:
                self.action_boost = {}
            self.urgency_strength = float(saved["urgency_strength"]) if "urgency_strength" in saved else 0.0
            self.train_loss_history = saved["train_loss_history"].tolist()
            self.validation_loss_history = saved["validation_loss_history"].tolist()
            self.X_mean_ = saved["mean"]
            self.X_std_ = saved["std"]
        return self
    
    # Softmax function to convert raw scores to probabilities
    @staticmethod
    def _softmax(scores):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def decision_function(self, X):
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        return np.dot(self._create_design_matrix(X), self.weights)
    
    def predict_proba(self, X):
        return self._softmax(self.decision_function(X))
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_action(self, observation):
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (12,):
            raise ValueError("Expected one observation with 12 values.")
        if self.weights is None:
            raise ValueError("Model has not been trained yet.")
        features = self._create_design_matrix(observation[None, :])
        action = int(self.classes_[np.argmax(features @ self.weights, axis=1)[0]])
        return action
    
    # See: https://archie-c.github.io/courses/deep-learning-aau/topics/02-classification/multiclass-classification-and-softmax/
    # You can adjust this if you want, there may be improvements to be found in the loss function.
    def cross_entropy_loss(self, y_true, y_pred, sample_weights=None):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0)
        per_sample_loss = -np.sum(y_true * np.log(y_pred), axis=1)
        if sample_weights is None:
            return np.mean(per_sample_loss)
        ##Change 3: weighted mean, stays on the same scale as the unweighted loss
        return np.sum(sample_weights * per_sample_loss) / np.sum(sample_weights)

    def _create_design_matrix(self, X, fit_normalisation=False):
        # Copy, so the caller's data isn't changed
        X = np.array(X, dtype=float)

        ##Change 4: the expert fires the engine when the braking needed to stop at the ground (vy² / 2h)
        # reaches what the engine can give (thrust / mass - g). A linear model can't build vy² / h by itself.
        # Uses the raw vy (ground speed), so this runs before Change 1.
        height = np.maximum(X[:, 1] - Y_GROUND, 0.1)  # no division by 0 at touchdown
        brake_needed = X[:, 3] ** 2 / (2 * height)
        brake_available = BURN_THRUST / (DRY_MASS + X[:, 6]) - G
        brake_ratio = brake_needed / brake_available
        # Clip, these explode right above the ground
        brake_needed = np.clip(brake_needed, 0, 3 * (BURN_THRUST / DRY_MASS - G))  # 3x the max possible braking
        brake_ratio = np.clip(brake_ratio, 0, 3)

        ##Change 5: with the engine off the wind turns the rocket (ω follows wind_x), with the engine on it doesn't.
        # A linear model can't switch wind_x on and off by itself, so it gets wind_x only while the engine is off.
        wind_x_engine_off = X[:, 9] * (X[:, 7] == 0)

        ##Change 6: the expert steers much more right above the ground (17 % of steps below 10 m, ~4 % higher up).
        # Normal θ stays, this extra copy gets boosted the closer the rocket is to the ground.
        theta_near_ground = X[:, 4] * np.exp(-height / BOOST_HEIGHT)

        ##Change 1: drop air_density (same info as y, corr -1.00) and use vy - wind_y (speed relative to the air)
        # Column order: x(0), y(1), vx(2), vy(3), theta(4), omega(5), fuel_mass(6), actual_thrust(7), actual_gimbal(8), wind_x(9), wind_y(10), air_density(11)
        X[:, 3] = X[:, 3] - X[:, 10]  # vy -> vy - wind_y
        X = np.delete(X, [10, 11], axis=1)  # drop wind_y and air_density

        X = np.column_stack((X, brake_needed, brake_available, brake_ratio, wind_x_engine_off, theta_near_ground))  # Change 4-6: 10 -> 15 features

        # Add the features before adding the bias term. For example, if you want to add polynomial features, you can do it here.
        
        # Bug fix: mean/std come from the training data only (set in fit). Recomputing them on every call gave
        # std = 0 for a single observation in predict_action -> all features 0 -> always the same action.
        if fit_normalisation:
            self.X_mean_ = np.mean(X, axis=0)
            self.X_std_ = np.std(X, axis=0)
        X = (X - self.X_mean_) / (self.X_std_ + 1e-12)  # Normalize features to have mean 0 and std 1
        
        X = np.column_stack((np.ones(X.shape[0]), X))  # Add bias term
        return X
    
    def _one_hot_encode(self, y):
        y = np.asarray(y)
        self.classes_, labels = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        return n_classes, np.eye(n_classes)[labels]
    
    # snapshot_epochs: for cross-validation, keeps a copy of the weights after these epochs,
    # so one run can be scored at several epoch counts instead of training again for each.
    def fit(self, train_X, train_y, validation_X=None, validation_y=None, snapshot_epochs=()):
        self.weight_snapshots = {}

        ##Change 8: urgency = braking needed / braking available (same as the Change 4 ratio), from the raw columns
        raw = np.array(train_X, dtype=float)
        height = np.maximum(raw[:, 1] - Y_GROUND, 0.1)
        urgency = np.clip(raw[:, 3] ** 2 / (2 * height) / (BURN_THRUST / (DRY_MASS + raw[:, 6]) - G), 0, 3)

        # Build the design matrix
        train_X = self._create_design_matrix(train_X, fit_normalisation=True)
        n_samples, n_features = train_X.shape
        
        # Convert the class labels to one-hot encoding
        n_classes, train_targets = self._one_hot_encode(train_y)

        ##Change 3: balanced weight = 1 / (n_classes * share of the action), softened by class_weight_power
        class_share = train_targets.mean(axis=0)
        class_weights = (1.0 / (n_classes * class_share)) ** self.class_weight_power
        ##Change 7: boost single actions on top
        for action, factor in self.action_boost.items():
            class_weights[self.classes_ == action] *= factor
        sample_weights = train_targets @ class_weights  # each row gets its action's weight
        ##Change 8: urgent rows count more, the harder the action brakes the more
        if self.urgency_strength > 0:
            row_thrust = train_targets @ THRUST_LEVEL[self.classes_.astype(int)]
            sample_weights = sample_weights * (1 + self.urgency_strength * urgency * row_thrust)

        # Initialise the weights
        self.weights = np.zeros(
            (n_features, n_classes)
        )
        
        # As we are using gradient descent, it's useful to keep track of the loss history for both training and validation
        # It can help us see if a model is overfitting or underfitting, and can also be used to implement early stopping if desired.
        if validation_X is not None and validation_y is not None:
            validation_X = self._create_design_matrix(validation_X)
            validation_y = np.asarray(validation_y)
            if not np.all(np.isin(validation_y, self.classes_)):
                raise ValueError("Validation labels must also appear in the training data.")
            validation_targets = (validation_y[:, None] == self.classes_).astype(float)
            
        # Reset the histories even if we aren't using the validation so its not carried over from a previous call.
        self.train_loss_history = []
        self.validation_loss_history = []
        
        # In this case, we are using the whole training set for each epoch.
        # If you are limited by memory you can modify this to use mini-batches.    
        for epoch in range(self.epochs):
            
            # Generate some probabilities for the training set
            probabilities = self._softmax(train_X @ self.weights)
            # Use the cross-entropy loss to evaluate the model's performance
            # Train loss is the weighted one plus the L2 penalty, that's what we optimise
            self.train_loss_history.append(
                self.cross_entropy_loss(train_targets, probabilities, sample_weights)
                + 0.5 * self.l2_lambda * np.sum(self.weights[1:] ** 2)
            )
            # If we have validation data, we can also evaluate the model's performance on that set
            # Validation loss stays unweighted so runs stay comparable
            if validation_X is not None and validation_y is not None:
                val_probabilities = self._softmax(validation_X @ self.weights)
                self.validation_loss_history.append(
                    self.cross_entropy_loss(validation_targets, val_probabilities)
                )
            progress = f"Epoch {epoch + 1}/{self.epochs} Train Loss: {self.train_loss_history[-1]:.4f}"
            if self.validation_loss_history:
                progress += f" Validation Loss: {self.validation_loss_history[-1]:.4f}"
            print(progress, end="\r")
            # Compute the update from the gradient of the loss function with respect to the weights
            error = probabilities - train_targets
            ##Change 3: weighted gradient
            dw = train_X.T @ (error * sample_weights[:, None]) / np.sum(sample_weights)
            dw[1:] += self.l2_lambda * self.weights[1:]  # L2, row 0 is the bias
            self.weights -= self.learning_rate * dw
            if epoch + 1 in snapshot_epochs:
                self.weight_snapshots[epoch + 1] = self.weights.copy()

        return self
