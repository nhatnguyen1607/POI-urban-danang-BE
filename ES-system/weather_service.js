const fetchWeatherData = async (lat, lng) => {
  const API_KEY = process.env.WEATHER_API_KEY;
  const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lng}&appid=${API_KEY}`;
  
  try {
    const response = await fetch(url);
    const data = await response.json();
    return {
      rain_1h: data.rain ? data.rain['1h'] : 0,
      description: data.weather && data.weather.length > 0 ? data.weather[0].main : 'Clear'
    };
  } catch (error) {
    return { rain_1h: 0, description: 'Clear' }; // Default
  }
};

module.exports = { fetchWeatherData };
