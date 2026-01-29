# Monisha Jetly - 21519830
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify API credentials.
client_id = '8fce2e48f32e4b69ba6f37a15e9bfeb8'
client_secret = '4d2509208766410b8ff84d9bbfa7f4b2'
redirect_uri = 'http://hand-gesture:8080/callback'

# Creating a SpotifyOAuth object to handle the authorization process.
sp_oauth = SpotifyOAuth(client_id=client_id,
                        client_secret=client_secret,
                        redirect_uri=redirect_uri,
                        scope='playlist-modify-private')

# Generating the authorization URL where the user needs to log in and authorize the application.
auth_url = sp_oauth.get_authorize_url()
print(f"Please visit this URL to authorize the application: {auth_url}")

# Prompting the user to enter the authorization code obtained from the redirect URL.
code = input("Enter the authorization code from the URL: ")

# Retrieving the token information using the authorization code.
token_info = sp_oauth.get_cached_token()
sp = spotipy.Spotify(auth=token_info['access_token'])

# Specifying the genre to search for tracks.
genre = 'rock'
# Searching for tracks based on the specified genre.
results = sp.search(q='genre:' + genre, type='track', limit=50)
# Extracting the track URIs from the search results.
track_uris = [track['uri'] for track in results['tracks']['items']]
# Creating a new public playlist.
playlist = sp.user_playlist_create(sp.me()['id'], 'My Rock Playlist', public=False)
# Adding the extracted tracks to the new playlist.
sp.playlist_add_items(playlist['id'], track_uris)
print('Playlist created successfully.')

