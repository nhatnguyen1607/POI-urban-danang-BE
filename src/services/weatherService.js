async function getForecast({ lat, lon }) {
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,precipitation,weather_code&hourly=precipitation_probability&timezone=Asia%2FBangkok&forecast_days=1`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Open-Meteo error: ${response.status}`);
  }
  const data = await response.json();
  const precipitation = data.current?.precipitation || 0;
  return {
    source: 'open-meteo',
    current: data.current,
    warning:
      precipitation > 0
        ? 'Dang co mua/luong mua, agent nen uu tien diem indoor hoac tuyen ngan.'
        : '',
  };
}

module.exports = {
  getForecast,
};
