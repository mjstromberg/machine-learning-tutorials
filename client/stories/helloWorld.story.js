import React from 'react';
import * as tf from '@tensorflow/tfjs';

export default class HelloWorld extends React.Component {
  constructor(props) {
    super(props);
    this.state = { isLoading: true, tensor: null };
  }

  componentDidMount() {
    initializeModel().then(tensor => {
      this.setState(state => {
        return { ...state, tensor };
      });
    });
  }

  renderTensor(tensor) {
    return (
      <React.Fragment>
        {JSON.stringify(tensor)}
      </React.Fragment>
    );
  }

  render () {
    return (
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        {this.state.tensor ? this.renderTensor(this.state.tensor) : 'Loading...'}
      </div>
    );
  }
}

function initializeModel() {
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Generate some synthetic data for training.
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  // Train the model using the data.
  return model.fit(xs, ys, {epochs: 10}).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
    return model.predict(tf.tensor2d([5], [1, 1]), { verbose: true }).data();
  });
}
