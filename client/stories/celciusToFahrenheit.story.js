import React from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import isNil from 'lodash/isNil';
import Big from 'big.js';

export default class CelciusToFahrenheit extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isTraining: true,
      model: null,
      prediction: null,
      inputValue: null
    };
  }

  componentDidMount() {
    // build visualization
    tfvis.visor().surface({
      name: 'Temperature Converter',
      tab: 'Data'
    });

    // initialize model
    trainModel().then(model => {
      this.setState(state => {
        return { ...state, isTraining: false, model };
      });
    });
  }

  handleChange = event => {
    const inputValue = Number(event.target.value);
    this.setState(state => {
      return { ...state, inputValue };
    })
  };

  handleClick = () => {
    this.setState(state => {
      const prediction = predict(state.model, state.inputValue);
      return { ...state, prediction };
    });
  }

  renderPlaceholder() {
    return 'Training...';
  }

  renderResults(prediction, inputValue) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: '10 10 0 0' }}>
          Prediction:
          {prediction}
        </div>
        <div style={{ padding: '10 10 0 0' }}>
          Calculated:
          {isNil(inputValue) ? '' : calculate(inputValue)}
        </div>
      </div>
    );
  }

  render () {
    const { isTraining, prediction, inputValue } = this.state;
    return (
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', flexDirection: 'row' }}>
          Celcius:
          <input disabled={isTraining} type="number" onChange={this.handleChange} />
          <button
            disabled={isNil(inputValue)}
            type="button"
            onClick={this.handleClick}
          >
            Predict
          </button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'row' }}>
          {isTraining
            ? this.renderPlaceholder()
            : this.renderResults(prediction, inputValue)}
        </div>
      </div>
    );
  }
}

const trainingData = {
  celcius: [-40, -10,  0,  8, 15, 22,  38],
  fahrenheit: [-40,  14, 32, 46, 59, 72, 100]
};

const testingData = {
  celcius: [-30, -12, 1, 17, 21, 40, 51],
  fahrenheit: [-22, 10.4, 33.8, 62.6, 69.8, 104, 123.8]
};

function calculate(celciusValue) {
  const big = Big(celciusValue);
  return Number(big.times(1.8).plus(32));
}

function predict(model, value) {
  // Use the model to do inference on a data point the model hasn't seen before:
  return model.predict(tf.tensor2d([value], [1, 1])).dataSync();
}

function trainModel() {
  // Define a model for linear regression.
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(1)
  });

  // Generate some synthetic data for training.
  const trainingCs = tf.tensor2d(trainingData.celcius, [ trainingData.celcius.length, 1 ]);
  const trainingFs = tf.tensor2d(trainingData.fahrenheit, [ trainingData.fahrenheit.length, 1 ]);

  // Generate some synthetic data for testing.
  const testingCs = tf.tensor2d(testingData.celcius, [ testingData.celcius.length, 1 ]);
  const testingFs = tf.tensor2d(testingData.fahrenheit, [ testingData.fahrenheit.length, 1 ]);

  // Train the model using the data.
  return model
    .fit(trainingCs, trainingFs, {
      batchSize: 64,
      callbacks: buildFitCallbacks(),
      epochs: 100,
      shuffle: true,
      validationData: [testingCs, testingFs]
    })
    .then(history => tf.tidy(() => model));
}

function buildFitCallbacks() {
  const metrics = ['loss', 'acc'];
  const container = {
    name: 'Training',
    tab: 'Training',
    styles: { height: '1000px' }
  };
  return tfvis.show.fitCallbacks(container, metrics);
}
