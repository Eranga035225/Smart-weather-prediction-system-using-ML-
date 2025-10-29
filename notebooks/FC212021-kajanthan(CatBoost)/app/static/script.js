
const cityCoords = {
  "Colombo": { x: 31.11, y: 74.48 }, "Gampaha": { x: 31.38, y: 68.75 }, "Kalutara": { x: 33.88, y: 81.51 },
  "Kandy": { x: 50.28, y: 63.80 }, "Matale": { x: 47.50, y: 55.47 }, "Nuwara Eliya": { x: 47.78, y: 72.40 },
  "Galle": { x: 37.78, y: 88.54 }, "Matara": { x: 45.56, y: 90.10 }, "Hambantota": { x: 57.50, y: 86.72 },
  "Jaffna": { x: 34.17, y: 10.68 }, "Kilinochchi": { x: 37.22, y: 18.75 }, "Mannar": { x: 34.17, y: 29.17 },
  "Vavuniya": { x: 43.61, y: 28.13 }, "Mullaitivu": { x: 46.67, y: 19.79 }, "Batticaloa": { x: 66.67, y: 51.04 },
  "Ampara": { x: 73.06, y: 66.41 }, "Trincomalee": { x: 56.94, y: 32.81 }, "Kurunegala": { x: 36.94, y: 54.69 },
  "Puttalam": { x: 29.72, y: 52.34 }, "Anuradhapura": { x: 43.06, y: 37.24 }, "Polonnaruwa": { x: 56.94, y: 47.14 },
  "Badulla": { x: 57.22, y: 71.35 }, "Moneragala": { x: 65.28, y: 74.48 }, "Ratnapura": { x: 45.83, y: 81.25 },
  "Kegalle": { x: 39.72, y: 69.01 }
};

const cityInput = document.getElementById("city");
const dateInput = document.getElementById("date");
const suggestionsList = document.getElementById("suggestions");
const cityMarker = document.getElementById("cityMarker");
const errorDiv = document.getElementById("error");

// Today's date
const today = new Date();
dateInput.value = today.toISOString().split('T')[0];

// Update city marker
function updateCityMarker(city) {
  if (cityCoords[city]) {
    const coords = cityCoords[city];
    cityMarker.style.left = `calc(${coords.x}% - 0.5rem)`;
    cityMarker.style.top = `calc(${coords.y}% - 0.5rem)`;
    cityMarker.style.display = "block";
  } else cityMarker.style.display = "none";
}

// City suggestions
cityInput.addEventListener("input", () => {
  const val = cityInput.value.trim().toLowerCase();
  suggestionsList.innerHTML = "";
  if (!val) return suggestionsList.classList.add("hidden");
  const filtered = Object.keys(cityCoords).filter(c => c.toLowerCase().includes(val));
  if (!filtered.length) return suggestionsList.classList.add("hidden");
  suggestionsList.classList.remove("hidden");
  filtered.forEach(city => {
    const li = document.createElement("li");
    li.textContent = city;
    li.className = "px-3 py-2 cursor-pointer hover:bg-gray-700";
    li.onclick = () => {
      cityInput.value = city;
      suggestionsList.classList.add("hidden");
      updateCityMarker(city);
    };
    suggestionsList.appendChild(li);
  });
});

document.addEventListener("click", e => {
  if (!cityInput.contains(e.target)) suggestionsList.classList.add("hidden");
});

// Fetch prediction
async function fetchPrediction(city, date) {
  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: new URLSearchParams({ city_name: city, date })
    });
    const data = await res.json();
    if (data.error) {
      showError(data.error);
      return;
    }
    errorDiv.classList.add("hidden");
    document.getElementById("temp_mean").textContent = `${data.prediction.temp_mean} °C`;
    document.getElementById("precip_sum").textContent = `${data.prediction.precip_sum} mm`;
    document.getElementById("rain_sum").textContent = `${data.prediction.rain_sum} mm`;
    document.getElementById("wind_speed_max").textContent = `${data.prediction.wind_speed_max} km/h`;
    document.getElementById("cityInfo").textContent = `${city} on ${date} || Location ID: ${data.location_id}`;
    updateCityMarker(city);
  } catch (err) {
    showError("Server error. Please try again later.");
  }
}

// Error display helper
function showError(msg) {
  errorDiv.textContent = msg;
  errorDiv.classList.remove("hidden");
}

// Handle form submit
document.getElementById("weatherForm").onsubmit = e => {
  e.preventDefault();
  const city = cityInput.value.trim();
  const date = dateInput.value.trim();

  if (!city || !cityCoords[city]) {
    showError("Please enter a valid city.");
    return;
  }
  if (!date) {
    showError("Please select a valid date.");
    return;
  }

  fetchPrediction(city, date);
};

// Initial load
fetchPrediction("Jaffna", dateInput.value);