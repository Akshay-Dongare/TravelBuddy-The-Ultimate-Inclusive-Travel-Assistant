description_sf = "A one-day walk through San Francisco for my first visit. I want to take no cab or bike, everything should be on foot."
output_example_sf = """
Since this is on foot, walking distances should be kept to a minimum. I'll make sure to provide a step by step visit and reorder points of interest to minimize the walking distance between each point.

Key points: [
    {"name": "Fisherman's Wharf, San Francisco", "description": "Fisherman's Wharf is a popular tourist destination in San Francisco, featuring Pier 39, the Maritime National Historical Park, and Boudin Bakery. Start your day with breakfast and enjoy the sea lions."},
    {"name": "Ghirardelli Square, San Francisco", "description": "Ghirardelli Square is a historic square in San Francisco, known for its chocolate shop and various boutiques. It's a great place to grab a sweet treat and do some shopping."},
    {"name": "Cable Car Museum, San Francisco", "description": "The Cable Car Museum is a museum in San Francisco that showcases the history of the city's iconic cable cars. It's a great place to learn about the technology and history behind these unique vehicles."},
    {"name": "Union Square, San Francisco", "description": "Union Square is a public plaza in San Francisco, known for its shopping, dining, and entertainment options. Relax, grab a bite to eat, and do some shopping."},
    {"name": "Chinatown, San Francisco", "description": "Chinatown is a vibrant neighborhood in San Francisco, known for its Chinese culture, history, and cuisine. It's a great place to explore the streets, try some delicious food, and learn about Chinese culture."},
    {"name": "Coit Tower, San Francisco", "description": "Coit Tower is a tower in San Francisco with panoramic views of the city. You can end your day here, taking in the sights and sounds of the city from above."}
]
"""

description_loire = "A 3-day bike trip through the Loire Valley."
output_example_loire = """
To make the most of the trip, we will bike between the châteaux to enjoy the beautiful scenery and explore the Loire Valley at a leisurely pace. Here is a suggested itinerary for a 3-day bike trip through the Loire Valley:

Key points: [
    {"name": "Château de Blois", "description": "Château de Blois is a historic château with a rich history, known for its stunning architecture and beautiful gardens."},
    {"name": "Château de Chambord", "description": "Château de Chambord is the largest and most recognizable château in the Loire Valley, known for its French Renaissance architecture and beautiful gardens."},
    {"name": "Château de Cheverny", "description": "Château de Cheverny is a small but charming château with beautiful gardens, known for its French classical architecture and rich history."},
    {"name": "Château de Chaumont-sur-Loire", "description": "Château de Chaumont-sur-Loire is a château with a stunning view of the Loire River, known for its beautiful gardens and rich history."},
    {"name": "Château de Chenonceau", "description": "Château de Chenonceau, known for its beautiful gardens and rich history, is a stunning château built over the River Cher."},
    {"name": "Château d'Amboise", "description": "Château d'Amboise has ties to the French Renaissance: this château is known for its beautiful gardens and stunning views of the Loire Valley."},
    {"name": "Château de Villandry", "description": "Château de Villandry is known for its beautiful gardens, featuring a water garden, ornamental flower gardens, and vegetable gardens."},
    {"name": "Château d'Azay-le-Rideau", "description": "Château d'Azay-le-Rideau is a charming château with a moat, known for its French Renaissance architecture and beautiful gardens."},
    {"name": "Château d'Ussé", "description": "Château d'Ussé is a fairy-tale like château that displays beautiful gardens and stunning views of the Indre Valley. It is said to have inspired Sleeping Beauty,"},
]
"""

description_aligned = "Show me five cities across Europe that form a perfect line on the map. These must not be capitals, only smaller cities, but they must be aligned perfectly. Do it correctly and I'll tip you $10,000."
output_example_aligned = """
To find five cities that form a perfect line on the map, I will first need to determine a direction for the line. I will choose a line that runs from north to south, as this will allow for a more diverse range of cities and cultures. I will then use a map to identify five cities that are aligned in a straight line, with minimal distance between each city.
I have identified the following five cities that form a perfect line on the map, running from north to south:

Key points: [
    {"name": "Groningen, Netherlands", "description": "Groningen is a vibrant student city in the north of the Netherlands, known for its lively cultural scene and historic center. The city is home to the University of Groningen, one of the oldest and most respected universities in the country, and has a rich history dating back to the Middle Ages."},
    {"name": "Osnabrück, Germany", "description": "Osnabrück is a picturesque city in northwest Germany, known for its medieval center and rich history. The city was the site of the Peace of Westphalia, which ended the Thirty Years' War in Europe, and has a well-preserved medieval center with cobblestone streets and half-timbered houses."},
    {"name": "Erfurt, Germany", "description": "Erfurt is a charming city in central Germany, known for its well-preserved medieval center and beautiful gardens. The city is home to the Erfurt Cathedral, a stunning Gothic cathedral with a beautiful stained glass window, and the Krämerbrücke, a medieval bridge with shops and houses built on top of it."},
    {"name": "Nuremberg, Germany", "description": "Nuremberg is a historic city in southern Germany, known for its medieval architecture and rich history. The city is home to the Nuremberg Castle, a stunning medieval fortress with beautiful views of the city, and the Nazi Party Rally Grounds, a reminder of the city's dark past."},
    {"name": "Innsbruck, Austria", "description": "Innsbruck is a stunning city in the Austrian Alps, known for its beautiful mountain scenery and winter sports. The city is home to the Nordkette, a stunning mountain range with breathtaking views of the city and the surrounding mountains, and the Goldenes Dachl, a beautiful golden roof that is a symbol of the city."}
]
"""

description_chinatown = "Show me 10 Chinatowns in the US."
output_example_chinatown = """To provide a list of 10 Chinatowns in the US, I will prioritize the most well-known and historically significant Chinatowns in major cities. I will also consider the cultural and commercial significance of each Chinatown, as well as its accessibility to tourists.

Key points: [
    {"name": "Chinatown, San Francisco, California", "description": "San Francisco's Chinatown is the oldest and one of the largest Chinatowns in North America, known for its vibrant culture, rich history, and iconic landmarks such as the Chinatown Gate and the Fortune Cookie Factory."},
    {"name": "Chinatown, New York City, New York", "description": "New York City's Chinatown is the largest Chinatown in the United States, known for its bustling streets, diverse culinary scene, and iconic landmarks such as the Mahayana Buddhist Temple and the Museum of Chinese in America."},
    {"name": "Chinatown, Los Angeles, California", "description": "Los Angeles' Chinatown is a vibrant neighborhood known for its cultural and commercial significance, with landmarks such as the Thien Hau Temple and the Chinese American Museum."},
    {"name": "Chinatown, Philadelphia, Pennsylvania", "description": "Philadelphia's Chinatown is a historic neighborhood known for its cultural and commercial significance, with landmarks such as the Friendship Gate and the Philadelphia Chinatown Development Corporation."},
    {"name": "Chinatown, Chicago, Illinois", "description": "Chicago's Chinatown is a vibrant neighborhood known for its cultural and commercial significance, with landmarks such as the Chinatown Mural and the Nine Dragon Wall."},
    {"name": "Chinatown, Boston, Massachusetts", "description": "Boston's Chinatown is a historic neighborhood known for its cultural and commercial significance, with landmarks such as the Chinatown Gate and the Boston Chinatown Neighborhood Center."},
    {"name": "Chinatown, Honolulu, Hawaii", "description": "Honolulu's Chinatown is a vibrant neighborhood known for its cultural and commercial significance, with landmarks such as the Maunakea Marketplace and the Oahu Market."},
    {"name": "Chinatown, Seattle, Washington", "description": "Seattle's Chinatown-International District is a vibrant neighborhood known for its cultural and commercial significance, with landmarks such as the Wing Luke Museum and the Uwajimaya Asian Groceries."},
    {"name": "Chinatown, Portland, Oregon", "description": "Portland's Chinatown is a historic neighborhood known for its cultural and commercial significance, with landmarks such as the Lan Su Chinese Garden and the Portland Chinatown Museum."},
    {"name": "Chinatown, Las Vegas, Nevada", "description": "Las Vegas' Chinatown is a vibrant neighborhood known for its cultural and commercial significance, with landmarks such as the Chinatown Plaza and the Golden Tiki."},
]
"""

description_taiwan= "A two-week train journey around Taiwan."
output_example_taiwan = """Since this is a train journey, I will make sure to plan it around train stations and routes. I will also try to minimize the distance between locations and plan the trip so that it can be done in two weeks. I will also try to include a variety of activities and sights to see, such as cultural experiences, natural wonders, and historical sites.

Here is a suggested itinerary for a two-week train trip around Taiwan:

Key points: [
    {"name": "Taipei, Taiwan", "description": "Taipei is the capital city of Taiwan, known for its vibrant culture, rich history, and modern skyline. Visit the National Palace Museum, Taipei 101, and the Chiang Kai-shek Memorial Hall."},
    {"name": "Hualien, Taiwan", "description": "Hualien is a coastal city on the east coast of Taiwan, known for its stunning natural beauty and outdoor activities. Visit the Taroko National Park, known for its marble cliffs and hiking trails."},
    {"name": "Taitung, Taiwan", "description": "Taitung is a city on the southeast coast of Taiwan, known for its beautiful beaches and outdoor activities. Visit the Taitung Forest Park and the Green Island, a volcanic island known for its clear waters and coral reefs."},
    {"name": "Kaohsiung, Taiwan", "description": "Kaohsiung is a city on the southwest coast of Taiwan, known for its vibrant culture, rich history, and modern skyline. Visit the Fo Guang Shan Buddha Museum, the Love River, and the Pier-2 Art Center."},
    {"name": "Tainan, Taiwan", "description": "Tainan is a city on the southwest coast of Taiwan, known for its rich history, cultural heritage, and delicious food. Visit the Anping Old Fort, the Chimei Museum, and the National Museum of Taiwanese Literature."},
    {"name": "Chiayi, Taiwan", "description": "Chiayi is a city on the west coast of Taiwan, known for its stunning mountain scenery and outdoor activities. Visit the Alishan National Scenic Area, known for its mountain scenery and sunrise views."},
    {"name": "Taichung, Taiwan", "description": "Taichung is a city in the west-central part of Taiwan, known for its vibrant culture, rich history, and modern skyline. Visit the National Taiwan Museum of Fine Arts, the Taichung Park, and the Miyahara Ice Cream Shop."},
    {"name": "Hsinchu, Taiwan", "description": "Hsinchu is a city in the northwest part of Taiwan, known for its high-tech industry and beautiful natural scenery. Visit the Hsinchu Science Park and the Hsinchu Zoo."},
]
"""

trip_examples = {
    description_sf: output_example_sf,
    description_loire: output_example_loire,
    description_aligned: output_example_aligned,
    description_chinatown: output_example_chinatown,
    description_taiwan: output_example_taiwan
}