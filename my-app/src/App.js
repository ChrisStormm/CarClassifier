import logo from './logo.svg';
import car from './2453352.png';
import './App.css';
import UploadAndDisplayImage from './Upload';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>CSE 455 Car Classifier</h1>
        <img src={car} className="App-logo" alt="logo" />
        <p>
        We will create a project that is able to accurately identify cars based on make and model
        from images. This could be useful in many different contexts, for example it could be used
        by law enforcement to help track down cars in an Amber Alert situation. It could also have
        a variety of usages in the automotive manufacturing, insurance, and urban planning industries.
        We will be using the Stanford cars data set, which contains 16,185 images belonging to 196 different
        classes of cars. The data is split into two datasets: training and testing. We will use a convolutional
        neural network to utilize and train on the car training dataset and help it eventually become 
        accurate enough to get high marks when run on the testing dataset.
        </p>
        <a
          className="App-link"
          href="https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset"
          target="_blank"
          rel="noopener noreferrer"
        >
          Data set Link
        </a>
        <UploadAndDisplayImage></UploadAndDisplayImage>
      </header>
    </div>
  );
}

export default App;
