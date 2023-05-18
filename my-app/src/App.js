import logo from './logo.svg';
import car from './car2.png';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>CSE 455 Car Classifier</h1>
        <img src={car} className="App-logo" alt="logo" />
        <p>
          Our Project:
        </p>
        <a
          className="App-link"
          href="https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset"
          target="_blank"
          rel="noopener noreferrer"
        >
          Data set Link
        </a>
      </header>
    </div>
  );
}

export default App;
