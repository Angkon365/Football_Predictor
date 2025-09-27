# --- Configuration for Desktop Testing ---
# This must be at the very top, before any other Kivy imports
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# -----------------------------------------

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.list import OneLineListItem
from kivy.network.urlrequest import UrlRequest
from kivy.clock import mainthread
from kivy.metrics import dp
import json

API_URL = "http://127.0.0.1:8000"
OTHER_LEAGUE = "Other (International/Cup)"

class FootballPredictorApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        screen = MDScreen()
        
        main_layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))

        # --- Title and Info Button ---
        title_box = MDBoxLayout(orientation='horizontal', adaptive_height=True, size_hint_y=None, height=dp(56))
        title_label = MDLabel(text="Football Predictor", halign="center", font_style="H4")
        info_button = MDIconButton(icon="information-outline", on_release=self.show_info_dialog)
        title_box.add_widget(title_label)
        title_box.add_widget(info_button)
        main_layout.add_widget(title_box)
        
        # --- Selections Layout ---
        selections_grid = MDGridLayout(cols=2, spacing=dp(15), adaptive_height=True)
        
        # --- Home Team Card ---
        home_card = MDCard(orientation='vertical', padding=dp(15), spacing=dp(10), size_hint_y=None, height=dp(160))
        home_card.add_widget(MDLabel(text="Home Team", font_style="H6", halign="center"))
        self.home_league_caller = self.create_league_caller('home')
        self.home_team_container = MDBoxLayout(orientation='vertical', adaptive_height=True)
        self.set_team_selector('home') # Initial setup
        home_card.add_widget(self.home_league_caller)
        home_card.add_widget(self.home_team_container)
        selections_grid.add_widget(home_card)

        # --- Away Team Card ---
        away_card = MDCard(orientation='vertical', padding=dp(15), spacing=dp(10), size_hint_y=None, height=dp(160))
        away_card.add_widget(MDLabel(text="Away Team", font_style="H6", halign="center"))
        self.away_league_caller = self.create_league_caller('away')
        self.away_team_container = MDBoxLayout(orientation='vertical', adaptive_height=True)
        self.set_team_selector('away') # Initial setup
        away_card.add_widget(self.away_league_caller)
        away_card.add_widget(self.away_team_container)
        selections_grid.add_widget(away_card)
        main_layout.add_widget(selections_grid)

        # --- Odds Card ---
        odds_card = MDCard(orientation='vertical', padding=dp(15), spacing=dp(10), size_hint_y=None, height=dp(240))
        odds_card.add_widget(MDLabel(text="Betting Odds", font_style="H6", halign="center"))
        self.odds_home_input = MDTextField(hint_text="Home Win Odds (e.g., 1.5)", input_filter="float")
        self.odds_draw_input = MDTextField(hint_text="Draw Odds (e.g., 3.0)", input_filter="float")
        self.odds_away_input = MDTextField(hint_text="Away Win Odds (e.g., 4.5)", input_filter="float")
        odds_card.add_widget(self.odds_home_input)
        odds_card.add_widget(self.odds_draw_input)
        odds_card.add_widget(self.odds_away_input)
        main_layout.add_widget(odds_card)

        # --- Prediction Button & Result ---
        main_layout.add_widget(self.create_prediction_button())
        self.create_result_display()
        main_layout.add_widget(self.result_card)

        screen.add_widget(main_layout)
        return screen

    def on_start(self):
        self.all_teams_data = {}
        self.all_teams_list = []
        self.home_league_menu = None
        self.away_league_menu = None
        self.home_team_menu = None
        self.away_team_menu = None
        self.home_suggestion_menu = None
        self.away_suggestion_menu = None
        self.fetch_leagues_and_teams()

    def show_info_dialog(self, instance):
        info_text = (
            "[b]Note on Team Availability:[/b]\n\n"
            "1. Some recently promoted teams are unavailable (e.g., Alverca, Kocaelispor, RAAL La Louviere, Pisa).\n\n"
            "2. Teams now in lower divisions can be selected if they are in our dataset."
        )
        if not hasattr(self, 'info_dialog'):
            self.info_dialog = MDDialog(title="Important Information", text=info_text, buttons=[MDRaisedButton(text="CLOSE", on_release=lambda x: self.info_dialog.dismiss())])
        self.info_dialog.open()

    def create_league_caller(self, role):
        caller = MDTextField(hint_text="Select League", readonly=True, icon_right="menu-down")
        caller.bind(on_touch_down=lambda instance, touch: self.handle_league_press(instance, touch, role))
        return caller

    def handle_league_press(self, instance, touch, role):
        if instance.collide_point(*touch.pos):
            menu = self.home_league_menu if role == 'home' else self.away_league_menu
            if menu: menu.open()

    def set_team_selector(self, role, league_name=None):
        container = self.home_team_container if role == 'home' else self.away_team_container
        container.clear_widgets()

        if league_name == OTHER_LEAGUE:
            # Create a text input for autocomplete
            team_input = MDTextField(hint_text=f"Type {role.capitalize()} Team Name")
            team_input.bind(text=lambda instance, value: self.show_suggestions(instance, value, role))
            container.add_widget(team_input)
            if role == 'home': self.home_team_selector = team_input
            else: self.away_team_selector = team_input
        else:
            # Create a dropdown caller
            team_caller = MDTextField(hint_text=f"Select {role.capitalize()} Team", readonly=True, icon_right="menu-down")
            team_caller.bind(on_touch_down=lambda instance, touch: self.handle_team_press(instance, touch, role))
            container.add_widget(team_caller)
            if role == 'home': self.home_team_selector = team_caller
            else: self.away_team_selector = team_caller
            # If a league is selected, populate the team dropdown
            if league_name:
                self.populate_team_dropdown(role, league_name)

    def handle_team_press(self, instance, touch, role):
        if instance.collide_point(*touch.pos):
            menu = self.home_team_menu if role == 'home' else self.away_team_menu
            if menu: menu.open()
    
    def create_prediction_button(self):
        return MDRaisedButton(text="GET PREDICTION", on_release=self.get_prediction, pos_hint={'center_x': 0.5}, size_hint_x=0.8, md_bg_color=self.theme_cls.primary_color)

    def create_result_display(self):
        self.result_card = MDCard(orientation='vertical', padding=dp(20), spacing=dp(10), size_hint_y=None, height=dp(140), md_bg_color=self.theme_cls.bg_dark)
        self.prediction_label = MDLabel(text="Prediction will appear here...", halign="center", font_style="H6")
        self.probability_label = MDLabel(text="", halign="center", theme_text_color="Secondary")
        self.warning_label = MDLabel(text="", halign="center", theme_text_color="Error", font_style="Caption")
        self.result_card.add_widget(self.prediction_label)
        self.result_card.add_widget(self.probability_label)
        self.result_card.add_widget(self.warning_label)

    # --- API & Data Logic ---
    def fetch_leagues_and_teams(self):
        UrlRequest(f"{API_URL}/leagues", on_success=self.on_data_success, on_failure=self.on_api_failure)

    @mainthread
    def on_data_success(self, request, result):
        self.all_teams_data = result.get('leagues', {})
        if not isinstance(self.all_teams_data, dict):
            self.on_api_failure(request, "Invalid data format from API.")
            return

        leagues = sorted(list(self.all_teams_data.keys())) + [OTHER_LEAGUE]
        home_menu_items = [{"viewclass": "OneLineListItem", "text": l, "on_release": lambda x=l: self.select_league(x, 'home')} for l in leagues]
        away_menu_items = [{"viewclass": "OneLineListItem", "text": l, "on_release": lambda x=l: self.select_league(x, 'away')} for l in leagues]
        
        self.home_league_menu = MDDropdownMenu(caller=self.home_league_caller, items=home_menu_items, width_mult=4)
        self.away_league_menu = MDDropdownMenu(caller=self.away_league_caller, items=away_menu_items, width_mult=4)

        all_teams_set = set()
        for team_list in self.all_teams_data.values():
            all_teams_set.update(team_list)
        self.all_teams_list = sorted(list(all_teams_set))

    def select_league(self, league_name, role):
        caller = self.home_league_caller if role == 'home' else self.away_league_caller
        caller.text = league_name
        self.set_team_selector(role, league_name)
        if role == 'home': self.home_league_menu.dismiss()
        else: self.away_league_menu.dismiss()

    def populate_team_dropdown(self, role, league_name):
        caller = self.home_team_selector if role == 'home' else self.away_team_selector
        teams = self.all_teams_data.get(league_name, [])
        items = [{"viewclass": "OneLineListItem", "text": t, "on_release": lambda x=t: self.select_team(x, role)} for t in teams]
        menu = MDDropdownMenu(caller=caller, items=items, width_mult=4)
        if role == 'home': self.home_team_menu = menu
        else: self.away_team_menu = menu

    def select_team(self, team_name, role):
        selector = self.home_team_selector if role == 'home' else self.away_team_selector
        selector.text = team_name
        menu = self.home_team_menu if role == 'home' else self.away_team_menu
        if menu: menu.dismiss()

    def show_suggestions(self, instance, text, role):
        # Dismiss previous suggestions
        sugg_menu = self.home_suggestion_menu if role == 'home' else self.away_suggestion_menu
        if sugg_menu: sugg_menu.dismiss()

        if text:
            suggestions = [t for t in self.all_teams_list if t.lower().startswith(text.lower())][:5] # Limit to 5
            if suggestions:
                items = [{"viewclass": "OneLineListItem", "text": s, "on_release": lambda x=s: self.select_suggestion(x, role)} for s in suggestions]
                menu = MDDropdownMenu(caller=instance, items=items, width_mult=4)
                menu.open()
                if role == 'home': self.home_suggestion_menu = menu
                else: self.away_suggestion_menu = menu

    def select_suggestion(self, team_name, role):
        selector = self.home_team_selector if role == 'home' else self.away_team_selector
        selector.text = team_name
        sugg_menu = self.home_suggestion_menu if role == 'home' else self.away_suggestion_menu
        if sugg_menu: sugg_menu.dismiss()

    def get_prediction(self, instance):
        self.warning_label.text = ""
        home_team = self.home_team_selector.text
        away_team = self.away_team_selector.text

        if not all([home_team, away_team, self.odds_home_input.text, self.odds_draw_input.text, self.odds_away_input.text]):
            self.prediction_label.text = "Error: All fields are required."
            return

        if home_team == away_team:
            self.prediction_label.text = "Error: Teams must be different."
            return

        payload = { "home_team": home_team, "away_team": away_team, "odds_home": float(self.odds_home_input.text), "odds_draw": float(self.odds_draw_input.text), "odds_away": float(self.odds_away_input.text) }
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        UrlRequest(f"{API_URL}/predict", req_body=json.dumps(payload), req_headers=headers, on_success=self.on_prediction_success, on_failure=self.on_api_failure)

    @mainthread
    def on_prediction_success(self, request, result):
        prediction = result.get('prediction', 'N/A')
        probs = result.get('probabilities', {})
        prob_text = f"H: {probs.get('home_win', 0)*100:.1f}% | D: {probs.get('draw', 0)*100:.1f}% | A: {probs.get('away_win', 0)*100:.1f}%"
        self.prediction_label.text = f"Predicted Outcome: {prediction}"
        self.probability_label.text = prob_text
        if self.home_league_caller.text == OTHER_LEAGUE or self.away_league_caller.text == OTHER_LEAGUE:
            self.warning_label.text = "Warning: H2H data unavailable. Prediction is less reliable."

    @mainthread
    def on_api_failure(self, request, error):
        self.prediction_label.text = "Error: Could not connect to API."
        self.probability_label.text = "Is the main.py server running?"
        print(f"API Error: {error}")

if __name__ == '__main__':
    FootballPredictorApp().run()

