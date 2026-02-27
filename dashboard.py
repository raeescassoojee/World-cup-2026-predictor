import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

model = joblib.load('xgb_model.pkl')
le = joblib.load('label_encoder.pkl')
df = pd.read_csv('df_processed.csv')
df['date'] = pd.to_datetime(df['date'])

tournament_weight = {
    'FIFA World Cup': 1.0, 'Copa América': 0.85, 'UEFA Euro': 0.85,
    'African Cup of Nations': 0.85, 'AFC Asian Cup': 0.85, 'CONCACAF Gold Cup': 0.80,
    'FIFA World Cup qualification': 0.75, 'UEFA Euro qualification': 0.70,
    'African Cup of Nations qualification': 0.70, 'AFC Asian Cup qualification': 0.70,
    'UEFA Nations League': 0.70, 'CONCACAF Nations League': 0.65, 'Friendly': 0.4
}
df['match_weight'] = df['tournament'].map(tournament_weight).fillna(0.5)

def get_team_stats(df, team, date, n_matches=30):
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(n_matches)
    if len(team_matches) < 5:
        return None
    stats = {}
    stats['matches_played'] = len(team_matches)
    wins = draws = losses = goals_scored = goals_conceded = 0
    weighted_wins = weighted_points = 0
    for _, match in team_matches.iterrows():
        weight = match['match_weight']
        if match['home_team'] == team:
            gs, gc = match['home_score'], match['away_score']
        else:
            gs, gc = match['away_score'], match['home_score']
        goals_scored += gs
        goals_conceded += gc
        if gs > gc:
            wins += 1; weighted_wins += weight; weighted_points += 3 * weight
        elif gs == gc:
            draws += 1; weighted_points += 1 * weight
        else:
            losses += 1
    n = len(team_matches)
    stats['win_rate'] = wins / n
    stats['draw_rate'] = draws / n
    stats['loss_rate'] = losses / n
    stats['wins'] = wins
    stats['draws'] = draws
    stats['losses'] = losses
    stats['avg_goals_scored'] = goals_scored / n
    stats['avg_goals_conceded'] = goals_conceded / n
    stats['goal_diff_avg'] = stats['avg_goals_scored'] - stats['avg_goals_conceded']
    stats['weighted_win_rate'] = weighted_wins / n
    stats['weighted_points_avg'] = weighted_points / n
    return stats

def get_head_to_head(df, team1, team2, date, n_matches=10):
    h2h = df[
        (((df['home_team'] == team1) & (df['away_team'] == team2)) |
         ((df['home_team'] == team2) & (df['away_team'] == team1))) &
        (df['date'] < date)
    ].tail(n_matches)
    if len(h2h) == 0:
        return {'h2h_win_rate': 0.5, 'h2h_goal_diff': 0, 'h2h_matches': 0, 'team1_wins': 0, 'team2_wins': 0, 'draws': 0, 'results': []}
    team1_wins = team2_wins = draws = 0
    team1_goal_diff = 0
    results = []
    for _, match in h2h.iterrows():
        if match['home_team'] == team1:
            diff = match['home_score'] - match['away_score']
            results.append({'date': match['date'], 'score': str(int(match['home_score']))+'-'+str(int(match['away_score'])),
                          'home': team1, 'away': team2, 'tournament': match['tournament']})
        else:
            diff = match['away_score'] - match['home_score']
            results.append({'date': match['date'], 'score': str(int(match['away_score']))+'-'+str(int(match['home_score'])),
                          'home': team2, 'away': team1, 'tournament': match['tournament']})
        team1_goal_diff += diff
        if diff > 0: team1_wins += 1
        elif diff == 0: draws += 1
        else: team2_wins += 1
    return {
        'h2h_win_rate': team1_wins / len(h2h), 'h2h_goal_diff': team1_goal_diff / len(h2h),
        'h2h_matches': len(h2h), 'team1_wins': team1_wins, 'team2_wins': team2_wins,
        'draws': draws, 'results': results
    }

def predict_match(home_team, away_team, neutral=False):
    date = df['date'].max()
    home_stats = get_team_stats(df, home_team, date)
    away_stats = get_team_stats(df, away_team, date)
    if home_stats is None or away_stats is None:
        return None, None, None, None
    h2h = get_head_to_head(df, home_team, away_team, date)
    match_features = pd.DataFrame([{
        'home_win_rate': home_stats['win_rate'], 'home_goals_scored': home_stats['avg_goals_scored'],
        'home_goals_conceded': home_stats['avg_goals_conceded'], 'home_goal_diff': home_stats['goal_diff_avg'],
        'home_weighted_points': home_stats['weighted_points_avg'], 'away_win_rate': away_stats['win_rate'],
        'away_goals_scored': away_stats['avg_goals_scored'], 'away_goals_conceded': away_stats['avg_goals_conceded'],
        'away_goal_diff': away_stats['goal_diff_avg'], 'away_weighted_points': away_stats['weighted_points_avg'],
        'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
        'goal_diff_diff': home_stats['goal_diff_avg'] - away_stats['goal_diff_avg'],
        'weighted_points_diff': home_stats['weighted_points_avg'] - away_stats['weighted_points_avg'],
        'goals_scored_diff': home_stats['avg_goals_scored'] - away_stats['avg_goals_scored'],
        'goals_conceded_diff': home_stats['avg_goals_conceded'] - away_stats['avg_goals_conceded'],
        'h2h_win_rate': h2h['h2h_win_rate'], 'h2h_goal_diff': h2h['h2h_goal_diff'],
        'h2h_matches': h2h['h2h_matches'], 'is_neutral': int(neutral), 'match_weight': 1.0
    }])
    probs = model.predict_proba(match_features)[0]
    prob_dict = {cls: prob for cls, prob in zip(le.classes_, probs)}
    return prob_dict, home_stats, away_stats, h2h

st.set_page_config(page_title="World Cup Predictor", page_icon="⚽", layout="wide")
st.title("⚽ World Cup Match Predictor")
st.markdown("*ML-powered match predictions using 150+ years of international football data*")

all_teams = sorted(list(set(df['home_team'].unique()) | set(df['away_team'].unique())))

st.sidebar.header("Match Setup")
home_team = st.sidebar.selectbox("Home Team", all_teams, index=all_teams.index('Brazil'))
away_team = st.sidebar.selectbox("Away Team", all_teams, index=all_teams.index('Argentina'))
neutral = st.sidebar.checkbox("Neutral Venue", value=True)
predict_btn = st.sidebar.button("Predict Match", use_container_width=True)

if predict_btn and home_team != away_team:
    probs, home_stats, away_stats, h2h = predict_match(home_team, away_team, neutral)
    if probs is None:
        st.error("Not enough data for one of these teams")
    else:
        st.header(f"{home_team} vs {away_team}")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{home_team} Win", f"{probs.get('Home Win', 0)*100:.1f}%")
        col2.metric("Draw", f"{probs.get('Draw', 0)*100:.1f}%")
        col3.metric(f"{away_team} Win", f"{probs.get('Away Win', 0)*100:.1f}%")

        fig_probs = go.Figure(go.Bar(
            x=[probs.get('Home Win', 0)*100, probs.get('Draw', 0)*100, probs.get('Away Win', 0)*100],
            y=[f'{home_team} Win', 'Draw', f'{away_team} Win'],
            orientation='h',
            marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
            text=[f"{probs.get('Home Win', 0)*100:.1f}%", f"{probs.get('Draw', 0)*100:.1f}%", f"{probs.get('Away Win', 0)*100:.1f}%"],
            textposition='auto'
        ))
        fig_probs.update_layout(title="Win Probability", xaxis_title="Probability %", height=250, margin=dict(t=40, b=20))
        st.plotly_chart(fig_probs, use_container_width=True)

        st.header("Team Comparison (Last 30 Matches)")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(home_team)
            s1, s2, s3 = st.columns(3)
            s1.metric("Win Rate", f"{home_stats['win_rate']*100:.0f}%")
            s2.metric("Goals/Game", f"{home_stats['avg_goals_scored']:.1f}")
            s3.metric("Conceded/Game", f"{home_stats['avg_goals_conceded']:.1f}")
            fig_home = go.Figure(go.Pie(
                values=[home_stats['wins'], home_stats['draws'], home_stats['losses']],
                labels=['Wins', 'Draws', 'Losses'],
                marker_colors=['#2ecc71', '#f39c12', '#e74c3c'], hole=0.4
            ))
            fig_home.update_layout(title=f"{home_team} Results", height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig_home, use_container_width=True)
        with col2:
            st.subheader(away_team)
            s1, s2, s3 = st.columns(3)
            s1.metric("Win Rate", f"{away_stats['win_rate']*100:.0f}%")
            s2.metric("Goals/Game", f"{away_stats['avg_goals_scored']:.1f}")
            s3.metric("Conceded/Game", f"{away_stats['avg_goals_conceded']:.1f}")
            fig_away = go.Figure(go.Pie(
                values=[away_stats['wins'], away_stats['draws'], away_stats['losses']],
                labels=['Wins', 'Draws', 'Losses'],
                marker_colors=['#2ecc71', '#f39c12', '#e74c3c'], hole=0.4
            ))
            fig_away.update_layout(title=f"{away_team} Results", height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig_away, use_container_width=True)

        categories = ['Win Rate', 'Goals Scored', 'Goal Diff', 'Weighted Points', 'H2H Win Rate']
        home_vals = [home_stats['win_rate'], home_stats['avg_goals_scored']/3,
                     (home_stats['goal_diff_avg']+2)/4, home_stats['weighted_points_avg']/3, h2h['h2h_win_rate']]
        away_vals = [away_stats['win_rate'], away_stats['avg_goals_scored']/3,
                     (away_stats['goal_diff_avg']+2)/4, away_stats['weighted_points_avg']/3, 1-h2h['h2h_win_rate']]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=home_vals + [home_vals[0]], theta=categories + [categories[0]],
                                             fill='toself', name=home_team, line_color='#2ecc71'))
        fig_radar.add_trace(go.Scatterpolar(r=away_vals + [away_vals[0]], theta=categories + [categories[0]],
                                             fill='toself', name=away_team, line_color='#e74c3c'))
        fig_radar.update_layout(title="Team Radar Comparison", polar=dict(radialaxis=dict(range=[0, 1])), height=400)
        st.plotly_chart(fig_radar, use_container_width=True)

        if h2h['h2h_matches'] > 0:
            st.header(f"Head to Head ({h2h['h2h_matches']} matches)")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{home_team} Wins", h2h['team1_wins'])
            c2.metric("Draws", h2h['draws'])
            c3.metric(f"{away_team} Wins", h2h['team2_wins'])
            if h2h['results']:
                h2h_df = pd.DataFrame(h2h['results'])
                h2h_df['date'] = pd.to_datetime(h2h_df['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(h2h_df[['date', 'home', 'away', 'score', 'tournament']], use_container_width=True, hide_index=True)

elif home_team == away_team:
    st.warning("Please select two different teams")
else:
    st.info("Select two teams and click Predict Match to get started!")

st.markdown("---")
st.caption("Built with XGBoost ML | Data: 49,071 international matches (1872-2026)")
