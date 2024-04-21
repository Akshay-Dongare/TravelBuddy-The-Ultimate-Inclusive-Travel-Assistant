import gradio as gr
from folium import Map
import numpy as np
from ast import literal_eval
import pandas as pd

import asyncio
from gradio_folium import Folium
import folium
from folium.plugins import Fullscreen
from huggingface_hub import InferenceClient
from geopy.geocoders import Nominatim
from collections import OrderedDict
from geopy.adapters import AioHTTPAdapter

import nest_asyncio
nest_asyncio.apply()

from examples import (
    description_sf,
    output_example_sf,
    description_loire,
    output_example_loire,
    description_taiwan,
    output_example_taiwan,
    trip_examples
)

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id, timeout=180)

end_sequence = "I hope that helps!"

def generate_key_points(text):
    prompt = f"""             
Please generate a set of key geographical points for the following description: {text}, as a json list of less than 10 dictionnaries with the following keys: 'name', 'description'.
Precise the full location in the 'name' if there is a possible ambiguity: for instance given that there are Chinatowns in several US cities, give the city name to disambiguate.
Generally try to minimize the distance between locations. Always think of the transportation means that you want to use, and the timing: morning, afternoon, where to sleep.
Only generate two sections: 'Thought:' provides your rationale for generating the points, then you list the locations in 'Key points:'.
Then generate '{end_sequence}' to indicate the end of the response.
For instance:
Description: {description_sf}
Thought: {output_example_sf}
{end_sequence}
Description: {description_loire}
Thought: {output_example_loire}
{end_sequence}
Now begin. You can make the descriptions a bit more verbose than in the examples.
Description: {text}
Thought:"""
    return llm_client.text_generation(prompt, max_new_tokens=2000, stream=True, stop_sequences=[end_sequence])


def parse_llm_output(output):
    rationale = "Thought: " + output.split("Key points:")[0]
    key_points = output.split("Key points:")[1]
    output = key_points.replace("    ", "").replace(end_sequence, "").strip()
    parsed_output = literal_eval(output)
    dataframe = pd.DataFrame.from_dict(parsed_output)
    return dataframe, rationale


class AsyncLRUCache:
    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    async def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    async def aset(self, key, value):
        self.set(key, value)

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

# Instantiate the cache
cache = AsyncLRUCache(maxsize=500)

preset_values = {
    "Fisherman's Wharf, San Francisco": {'lat': 37.808332, 'lon': -122.415715},
    'Ghirardelli Square, San Francisco': {'lat': 37.80587075, 'lon': -122.42294914207058},
    'Cable Car Museum, San Francisco': {'lat': 37.79476015, 'lon': -122.41185284314184},
    'Union Square, San Francisco': {'lat': 37.7875138, 'lon': -122.407159},
    'Chinatown, San Francisco': {'lat': 37.7943011, 'lon': -122.4063757},
    'Coit Tower, San Francisco': {'lat': 37.80237905, 'lon': -122.40583435461313},
    'Chinatown, San Francisco, California': {'lat': 37.7943011, 'lon': -122.4063757},
    'Chinatown, New York City, New York': {'lat': 40.7164913, 'lon': -73.9962504},
    'Chinatown, Los Angeles, California': {'lat': 34.0638402, 'lon': -118.2358676},
    'Chinatown, Philadelphia, Pennsylvania': {'lat': 39.9534461, 'lon': -75.1546218},
    'Chinatown, Chicago, Illinois': {'lat': 41.8516579, 'lon': -87.6331383},
    'Chinatown, Boston, Massachusetts': {'lat': 42.3513291, 'lon': -71.0626228},
    'Chinatown, Honolulu, Hawaii': {'lat': 21.3129031, 'lon': -157.8628003},
    'Chinatown, Seattle, Washington': {'lat': 47.5980601, 'lon': -122.3245246},
    'Chinatown, Portland, Oregon': {'lat': 45.5251092, 'lon': -122.6744481},
    'Chinatown, Las Vegas, Nevada': {'lat': 36.2823279, 'lon': -115.3310655},
    'Taipei, Taiwan': {'lat': 25.0375198, 'lon': 121.5636796},
    'Hualien, Taiwan': {'lat': 23.9913421, 'lon': 121.6197276},
    'Taitung, Taiwan': {'lat': 22.7553667, 'lon': 121.1506},
    'Kaohsiung, Taiwan': {'lat': 22.6203348, 'lon': 120.3120375},
    'Tainan, Taiwan': {'lat': 22.9912348, 'lon': 120.184982},
    'Chiayi, Taiwan': {'lat': 23.4591664, 'lon': 120.2930004},
    'Taichung, Taiwan': {'lat': 24.163162, 'lon': 120.6478282},
    'Hsinchu, Taiwan': {'lat': 24.8066333, 'lon': 120.9686833},
    'Château de Blois': {'lat': 47.650198, 'lon': 1.426256515186913}, 
    'Château de Chambord': {'lat': 47.61606945, 'lon': 1.5170501827851928},
    'Château de Cheverny': {'lat': 47.50023105, 'lon': 1.4580181089595223},
    'Château de Chaumont-sur-Loire': {'lat': 47.479146, 'lon': 1.181523652578578},
    'Château de Chenonceau': {'lat': 47.32461905, 'lon': 1.070403778072624},
    "Château d'Amboise": {'lat': 47.41362905, 'lon': 0.9859718927689629},
    'Château de Villandry': {'lat': 47.34056095, 'lon': 0.5146088880523084},
    "Château d'Azay-le-Rideau": {'lat': 47.25904985, 'lon': 0.465756301165524},
    "Château d'Ussé": {'lat': 47.249807, 'lon': 0.2909891848913879},
    'Groningen, Netherlands': {'lat': 53.2190652, 'lon': 6.5680077},
    'Osnabrück, Germany': {'lat': 52.37265095, 'lon': 8.161049572938472},
    'Erfurt, Germany': {'lat': 50.9777974, 'lon': 11.0287364},
    'Nuremberg, Germany': {'lat': 49.453872, 'lon': 11.077298},
    'Innsbruck, Austria': {'lat': 47.2654296, 'lon': 11.3927685},
    'Embarcadero, San Francisco': {'lat': 37.7928637, 'lon': -122.396912},
    'Pier 39, San Francisco': {'lat': 37.808703, 'lon': -122.410116},
    'Palace of Fine Arts, San Francisco': {'lat': 37.80291855, 'lon': -122.44840286435331},
    'Crissy Field, San Francisco': {'lat': 37.80459605, 'lon': -122.4666072420753},
    'Golden Gate Bridge, San Francisco': {'lat': 37.8302731, 'lon': -122.4798443},
    'Fort Point National Historic Site, San Francisco': {'lat': 37.81045495, 'lon': -122.47713831312802},
    'Presidio of San Francisco': {'lat': 37.798745600000004, 'lon': -122.46458892410745}
}
for key, value in preset_values.items():
    cache.set(key, value)


async def geocode_address(address):
    # Check if the result is in cache
    cached_location = await cache.get(address)
    if cached_location:
        return cached_location

    # If not in cache, perform the geolocation request
    async with Nominatim(
        user_agent="HF-trip-planner",
        adapter_factory=AioHTTPAdapter,
    ) as geolocator:
        location = await geolocator.geocode(address, timeout=10)
        if location:
            coords = {'lat': location.latitude, "lon": location.longitude}
            # Save the result in cache for future use
            await cache.aset(address, coords)
            return coords
        return None

async def ageocode_addresses(addresses):
    tasks = [geocode_address(address) for address in addresses]
    locations = await asyncio.gather(*tasks)
    return locations

def geocode_addresses(addresses):
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(ageocode_addresses(addresses))
    return result


def create_map_from_markers(dataframe):
    coordinates = geocode_addresses(dataframe["name"])
    print({name: coordinates[i] for i, name in enumerate(dataframe["name"].to_list())})
    dataframe["lat"] = [coords["lat"] if coords else None for coords in coordinates]
    dataframe["lon"] = [coords["lon"] if coords else None for coords in coordinates]

    f_map = Map(
        location=[dataframe["lat"].mean(), dataframe["lon"].mean()],
        zoom_start=5,
        tiles=folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Maps",
            overlay=True,
            control=True,
        ),
    )
    for _, row in dataframe.iterrows():
        if np.isnan(row["lat"]) or np.isnan(row["lon"]):
            continue
        popup_message = f"<h4 style='color: #d53e2a;'>{row['name']}</h4><p style='font-weight:500'>{row['description']}</p>"
        popup_message += f"<a href='https://www.google.com/search?q={row['name']}' target='_blank'><b>Learn more about {row['name'].split(',')[0]}</b></a>"

        marker = folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_message, max_width=200),
            icon=folium.Icon(color="yellow", icon="fa-circle-dot", prefix='fa'),
        )
        marker.add_to(f_map),
    
    Fullscreen(position='topright', title='Expand me', title_cancel='Exit me', force_separate_button=True).add_to(f_map)

    bounds = [[row["lat"], row["lon"]] for _, row in dataframe.iterrows()]
    f_map.fit_bounds(bounds, padding=(100, 100))
    return f_map


def run_display(text):
    current_output = ""
    for output in generate_key_points(text):
        current_output += output
        yield None, "```text\n" + current_output + "\n```"
    current_output = current_output.replace("</s>", "")
    dataframe, _ = parse_llm_output(current_output)
    map = create_map_from_markers(dataframe)
    yield map, "```text\n" + current_output + "\n```"


def select_example(choice):
    output = trip_examples[choice]
    dataframe, _ = parse_llm_output(output)
    map = create_map_from_markers(dataframe)
    return choice, map, "```text\n" + output + "\n```"


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.yellow,
        secondary_hue=gr.themes.colors.blue,
    )
) as demo:
    gr.Markdown("# 🗺️ AI Travel Planner 🏕️\nThis personal travel planner is based on Mixtral-8x7B, called through the Hugging Face API. Describe your ideal trip below, and let our AI assistant guide you!\n Beware that the model does not really have access to train or plane schedules, it is relying on general world knowledge for its propositions.")
    text = gr.Textbox(
        label="Describe your ideal trip:",
        value=description_taiwan,
    )
    button = gr.Button("Generate trip!")
            
    gr.Markdown("### LLM Output 👇")

    example_dataframe, _ = parse_llm_output(output_example_taiwan)
    display_thoughts = gr.Markdown("```text\n" + output_example_sf + "\n```")

    gr.Markdown("_Click the markers on the map map to display information about the places._")
    # Get initial map
    starting_map = create_map_from_markers(example_dataframe)
    map = Folium(value=starting_map, height=600, label="Chosen locations")

    # Trip examples
    clickable_examples = gr.Dropdown(choices=trip_examples.keys(), label="Try another example:", value=description_taiwan)

    # Dynamics
    button.click(run_display, inputs=[text], outputs=[map, display_thoughts])
    clickable_examples.input(
        select_example, clickable_examples, outputs=[text, map, display_thoughts]
    )

if __name__ == "__main__":
    demo.launch()