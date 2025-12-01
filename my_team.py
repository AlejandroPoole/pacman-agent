# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveMinimaxAgent', second='DefensiveMinimaxAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


class OffensiveMinimaxAgent(ReflexCaptureAgent):
    """
    An offensive agent that uses minimax to choose actions.
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.depth = 1  # Set search depth
        self.initial_target_reached = False  # Track if we've reached bottom
    
    def register_initial_state(self, game_state):
        """Called at the start of the game"""
        super().register_initial_state(game_state)
        self.initial_target_reached = False
        
        # Set target to bottom of the maze
        height = game_state.data.layout.height
        width = game_state.data.layout.width
        
        # Find the bottom-most valid position on enemy side
        if self.red:
            target_x = width // 2  # Just across boundary
        else:
            target_x = (width // 2) - 1  # Just across boundary
        
        # Find lowest y position that's valid (not a wall)
        for y in range(height):
            if not game_state.has_wall(target_x, y):
                self.initial_bottom_target = (target_x, y)
                break
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Food collection priority
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        
        # Distance to nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # Check if we're carrying food
        food_carrying = my_state.num_carrying
        
        # If carrying food, prioritize returning home
        if food_carrying > 0:
            features['carrying_food'] = food_carrying
            
            # Check if we've actually returned home (became a ghost again)
            if not my_state.is_pacman:
                # Successfully returned - give huge bonus
                features['returned_home'] = 1
            else:
                # Still on enemy side - just move toward our side (negative x for red, positive x for blue)
                if self.red:
                    # Red team: want to decrease x (move left)
                    features['x_position'] = my_pos[0]
                else:
                    # Blue team: want to increase x (move right)
                    features['x_position'] = -my_pos[0]
        
        # Check for nearby enemy ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        
        if len(ghosts) > 0 and my_state.is_pacman:
            ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            min_ghost_dist = min(ghost_distances)
            
            # Ghost danger increases priority to return home
            if min_ghost_dist <= 1:
                features['ghost_nearby'] = 1
                features['ghost_distance'] = min_ghost_dist
        
        # Penalize reversing direction to avoid getting stuck
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        
        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        
        # If carrying food, prioritize returning home heavily
        if my_state.num_carrying > 0:
            return {
                'successor_score': -1000,      # AVOID eating more food when carrying
                'distance_to_food': 0,         # Don't care about food distance
                'carrying_food': 100,           
                'returned_home': 5000,         # MASSIVE bonus for crossing home
                'x_position': -500,            # Very strong incentive to move toward our side
                'ghost_nearby': -300,          
                'ghost_distance': 2,           
                'reverse': -2                  # Very light penalty when escaping
            }
        else:
            # Not carrying food - focus on collecting
            return {
                'successor_score': 200,        # HIGH priority for eating food
                'distance_to_food': -2,        # Strong incentive to approach food
                'carrying_food': 0,
                'returned_home': 0,
                'x_position': 0,
                'ghost_nearby': -500,
                'ghost_distance': 5,
                'reverse': -10                 # Moderate penalty to avoid oscillation
            }
    
    def defensive_action(self, game_state):
        """
        Defensive behavior when winning - patrol the boundary
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        actions = game_state.get_legal_actions(self.index)
        
        # Remove STOP to keep patrolling
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)
        
        values = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_state = successor.get_agent_state(self.index)
            successor_pos = successor_state.get_position()
            
            score = 0
            
            # Check for invaders
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            
            if len(invaders) > 0:
                # Chase invaders
                invader_distances = [self.get_maze_distance(successor_pos, inv.get_position()) for inv in invaders]
                min_invader_dist = min(invader_distances)
                score -= min_invader_dist * 1000
            else:
                # Patrol near boundary
                boundary_x = game_state.data.layout.width // 2
                if self.red:
                    boundary_x -= 1
                
                distance_to_boundary = abs(successor_pos[0] - boundary_x)
                score -= distance_to_boundary * 100
            
            # Stay on our side (be a ghost)
            if not successor_state.is_pacman:
                score += 500
            
            values.append(score)
        
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return best_actions[0]
    
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
    
    def choose_action(self, game_state):
        """
        Returns the minimax action using minimax algorithm
        """
        my_state = game_state.get_agent_state(self.index)
        current_score = self.get_score(game_state)
        my_pos = my_state.get_position()
        time_left = game_state.data.timeleft
        
        # Check if both agents have food - if so, both should return home quickly
        our_team = [self.index] + [i for i in self.get_team(game_state) if i != self.index]
        team_states = [game_state.get_agent_state(i) for i in our_team]
        both_have_food = all(state.num_carrying > 0 for state in team_states)
        
        # DESPERATION MODE: If game is almost over (< 50 moves) and we're losing, go reckless for food
        desperate = time_left < 50 and current_score < 0
        
        # INITIAL STRATEGY: Go to bottom of maze at game start
        if not self.initial_target_reached and hasattr(self, 'initial_bottom_target'):
            dist_to_target = self.get_maze_distance(my_pos, self.initial_bottom_target)
            
            # If we're close to the target or have collected food, consider target reached
            if dist_to_target <= 2 or my_state.num_carrying > 0:
                self.initial_target_reached = True
            else:
                # Head to bottom target
                actions = game_state.get_legal_actions(self.index)
                values = []
                
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    successor_state = successor.get_agent_state(self.index)
                    successor_pos = successor_state.get_position()
                    
                    # Distance to bottom target
                    dist = self.get_maze_distance(successor_pos, self.initial_bottom_target)
                    score = -dist * 1000
                    
                    # Avoid stopping
                    if action == Directions.STOP:
                        score -= 500
                    
                    values.append(score)
                
                max_value = max(values)
                best_actions = [a for a, v in zip(actions, values) if v == max_value]
                return best_actions[0]
        
        # Check if we can kill nearby enemies
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        
        # If we're a ghost, chase nearby enemy pacmen
        if not my_state.is_pacman:
            enemy_pacmen = [a for a in enemies if a.is_pacman and a.get_position() is not None]
            
            if len(enemy_pacmen) > 0:
                pacman_distances = [(self.get_maze_distance(my_pos, p.get_position()), p) for p in enemy_pacmen]
                min_dist, closest_pacman = min(pacman_distances, key=lambda x: x[0])
                
                if min_dist <= 2:
                    # Chase the enemy pacman
                    actions = game_state.get_legal_actions(self.index)
                    values = []
                    
                    for action in actions:
                        successor = self.get_successor(game_state, action)
                        successor_state = successor.get_agent_state(self.index)
                        successor_pos = successor_state.get_position()
                        
                        # Distance to enemy pacman
                        dist_to_enemy = self.get_maze_distance(successor_pos, closest_pacman.get_position())
                        score = -dist_to_enemy * 10000  # Very high priority
                        
                        values.append(score)
                    
                    max_value = max(values)
                    best_actions = [a for a, v in zip(actions, values) if v == max_value]
                    return best_actions[0]
        
        # If we're a pacman, chase nearby scared ghosts
        else:
            scared_ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer > 0]
            
            if len(scared_ghosts) > 0:
                ghost_distances = [(self.get_maze_distance(my_pos, g.get_position()), g) for g in scared_ghosts]
                min_dist, closest_ghost = min(ghost_distances, key=lambda x: x[0])
                
                if min_dist <= 2:
                    # Chase the scared ghost
                    actions = game_state.get_legal_actions(self.index)
                    values = []
                    
                    for action in actions:
                        successor = self.get_successor(game_state, action)
                        successor_state = successor.get_agent_state(self.index)
                        successor_pos = successor_state.get_position()
                        
                        # Distance to scared ghost
                        dist_to_ghost = self.get_maze_distance(successor_pos, closest_ghost.get_position())
                        score = -dist_to_ghost * 10000  # Very high priority
                        
                        values.append(score)
                    
                    max_value = max(values)
                    best_actions = [a for a, v in zip(actions, values) if v == max_value]
                    return best_actions[0]
        
        # If winning by 3+ points, switch to defensive mode (unless desperate)
        if current_score >= 3 and not desperate:
            return self.defensive_action(game_state)
        
        # If carrying food, use simpler greedy approach to avoid oscillation
        # URGENT: If both agents have food, return home immediately
        if my_state.num_carrying > 0:
            actions = game_state.get_legal_actions(self.index)
            
            # Remove STOP action to prevent getting stuck
            if Directions.STOP in actions and len(actions) > 1:
                actions.remove(Directions.STOP)
            
            values = []
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_state = successor.get_agent_state(self.index)
                successor_pos = successor_state.get_position()
                
                # Check for ghosts
                enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
                ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
                
                score = 0
                
                # URGENT: If both agents have food, prioritize getting home even more
                home_urgency = 200000 if both_have_food else 100000
                
                # Massive bonus if this action takes us home (becomes ghost)
                if not successor_state.is_pacman:
                    score += home_urgency  # Huge score for crossing home
                else:
                    # Look ahead 2 steps to see if we make progress
                    current_pos = my_state.get_position()
                    current_x = current_pos[0]
                    successor_x = successor_pos[0]
                    
                    # Check if we're making progress toward home
                    if self.red:
                        # Red wants to decrease x (go left)
                        x_progress = current_x - successor_x  # Positive if moving left
                    else:
                        # Blue wants to increase x (go right)
                        x_progress = successor_x - current_x  # Positive if moving right
                    
                    # Reward actual progress toward home
                    score += x_progress * 5000
                    
                    # Also consider final x position (but less important than progress)
                    if self.red:
                        score += -successor_x * 100
                    else:
                        score += successor_x * 100
                    
                    # Penalize reversing direction heavily to avoid oscillation
                    current_direction = game_state.get_agent_state(self.index).configuration.direction
                    if action == Directions.REVERSE[current_direction]:
                        score -= 2000
                    
                    # Dynamic ghost avoidance based on score and desperation
                    # When desperate (losing + low time): NO FEAR except death
                    # When losing: only fear ghosts at distance <= 1 (very reckless)
                    # Score 1: fear at distance <= 2
                    # Score 2: fear at distance <= 3
                    # Score 3+: fear at distance <= 5 (maximum caution)
                    if len(ghosts) > 0:
                        ghost_distances = [self.get_maze_distance(successor_pos, g.get_position()) for g in ghosts]
                        min_ghost_dist = min(ghost_distances)
                        
                        # Determine fear threshold based on score and desperation
                        if desperate:
                            fear_distance = 0  # ZERO FEAR - only avoid actual death
                        elif current_score <= 0:
                            fear_distance = 1  # Very reckless - only avoid immediate contact
                        elif current_score == 1:
                            fear_distance = 2
                        elif current_score == 2:
                            fear_distance = 3
                        else:
                            fear_distance = 5  # Maximum caution when winning by 3+
                        
                        # Apply penalties only within fear distance
                        if min_ghost_dist == 0:
                            score -= 1000000  # Death - always avoid
                        elif min_ghost_dist <= fear_distance:
                            # Scale penalty based on distance
                            if min_ghost_dist == 1:
                                score -= 50000
                            elif min_ghost_dist == 2:
                                score -= 10000
                            elif min_ghost_dist == 3:
                                score -= 1000
                            elif min_ghost_dist == 4:
                                score -= 500
                            elif min_ghost_dist == 5:
                                score -= 100
                        
                
                values.append(score)
            
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            return best_actions[0]  # Take first best action
        
        # Not carrying food - use full minimax
        def value(state, agent, depth):
            if state.is_over() or depth == self.depth:
                return self.evaluation_function(state), None
            
            # Check if agent position is valid
            agent_state = state.get_agent_state(agent)
            if agent_state.get_position() is None:
                # Agent was eaten, skip to next agent
                next_agent = self.get_next_agent(state, agent)
                next_depth = depth + 1 if next_agent == self.index else depth
                return value(state, next_agent, next_depth)
            
            legal_actions = state.get_legal_actions(agent)
            
            # Handle case where agent has no legal actions
            if not legal_actions or len(legal_actions) == 0:
                return self.evaluation_function(state), None
            
            if agent == self.index:  # max (our agent)
                max_score = -100000
                max_action = None
                for action in legal_actions:
                    successor = state.generate_successor(agent, action)
                    score, _ = value(successor, self.get_next_agent(state, agent), depth)
                    if score > max_score:
                        max_score = score
                        max_action = action
                return max_score, max_action
            
            else:  # min (opponent or teammate)
                min_score = 100000
                min_action = None
                next_agent = self.get_next_agent(state, agent)
                next_depth = depth + 1 if next_agent == self.index else depth
                
                for action in legal_actions:
                    successor = state.generate_successor(agent, action)
                    score, _ = value(successor, next_agent, next_depth)
                    if score < min_score:
                        min_score = score
                        min_action = action
                
                return min_score, min_action

        _, best_action = value(game_state, self.index, 0)
        return best_action if best_action else Directions.STOP

    def get_next_agent(self, game_state, current_agent):
        """Get the next agent in turn order"""
        next_agent = current_agent + 1
        if next_agent >= game_state.get_num_agents():
            next_agent = 0
        return next_agent

    def evaluation_function(self, game_state):
        """
        Evaluation function using get_features and get_weights
        """
        features = self.get_features(game_state, Directions.STOP)
        weights = self.get_weights(game_state, Directions.STOP)
        return features * weights


class DefensiveMinimaxAgent(CaptureAgent):
    """
    A defensive agent that uses minimax and guards high-value areas
    """
    
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.depth = 2
        self.start = None
        self.food_clusters = []
    
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        
        # Calculate food clusters on our side to defend
        self.calculate_defense_positions(game_state)
    
    def calculate_defense_positions(self, game_state):
        """Find positions with high food density on our side"""
        # Get all food we're defending
        food_list = self.get_food_you_are_defending(game_state).as_list()
        
        if len(food_list) == 0:
            self.defense_center = self.start
            self.food_clusters = []
            return
        
        # Find center of mass of our food
        avg_x = sum([f[0] for f in food_list]) / len(food_list)
        avg_y = sum([f[1] for f in food_list]) / len(food_list)
        
        # Find the closest valid (non-wall) position to the center
        best_pos = None
        best_dist = float('inf')
        
        for food in food_list:
            dist = abs(food[0] - avg_x) + abs(food[1] - avg_y)
            if dist < best_dist:
                best_dist = dist
                best_pos = food
        
        self.defense_center = best_pos if best_pos else self.start
        
        # Identify food clusters (groups of 3+ food pellets close together)
        self.food_clusters = []
        for food in food_list:
            nearby_food = [f for f in food_list if self.get_maze_distance(food, f) <= 3]
            if len(nearby_food) >= 3:
                # This is a cluster
                cluster_center_x = sum([f[0] for f in nearby_food]) / len(nearby_food)
                cluster_center_y = sum([f[1] for f in nearby_food]) / len(nearby_food)
                
                # Find closest food position to this cluster center
                closest_food = min(nearby_food, key=lambda f: abs(f[0] - cluster_center_x) + abs(f[1] - cluster_center_y))
                
                # Avoid duplicates
                if closest_food not in self.food_clusters:
                    self.food_clusters.append(closest_food)
        
        # Keep only top 3 largest clusters
        self.food_clusters = self.food_clusters[:3]
    
    def get_successor(self, game_state, action):
        """Finds the next successor"""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor
    
    def choose_action(self, game_state):
        """Returns the minimax action"""
        my_state = game_state.get_agent_state(self.index)
        current_score = self.get_score(game_state)
        time_left = game_state.data.timeleft
        
        # Check if both agents have food - if so, both should return home quickly
        our_team = [self.index] + [i for i in self.get_team(game_state) if i != self.index]
        team_states = [game_state.get_agent_state(i) for i in our_team]
        both_have_food = all(state.num_carrying > 0 for state in team_states)
        
        # DESPERATION MODE: If game is almost over (< 50 moves) and we're losing, become offensive
        desperate = time_left < 50 and current_score < 0
        
        # If both agents have food, return home immediately
        if my_state.num_carrying > 0 and both_have_food:
            actions = game_state.get_legal_actions(self.index)
            if Directions.STOP in actions and len(actions) > 1:
                actions.remove(Directions.STOP)
            
            values = []
            my_pos = my_state.get_position()
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_state = successor.get_agent_state(self.index)
                successor_pos = successor_state.get_position()
                
                score = 0
                if not successor_state.is_pacman:
                    score += 200000  # URGENT: Get home
                else:
                    # Progress toward home
                    current_x = my_pos[0]
                    successor_x = successor_pos[0]
                    if self.red:
                        x_progress = current_x - successor_x
                    else:
                        x_progress = successor_x - current_x
                    score += x_progress * 10000
                
                values.append(score)
            
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            return best_actions[0]
        
        # If desperate and we're on enemy side, grab food aggressively
        if desperate and my_state.is_pacman:
            actions = game_state.get_legal_actions(self.index)
            values = []
            my_pos = my_state.get_position()
            food_list = self.get_food(game_state).as_list()
            
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_state = successor.get_agent_state(self.index)
                successor_pos = successor_state.get_position()
                
                # Prioritize getting food
                score = successor.get_score() * 1000
                
                if len(food_list) > 0:
                    min_food_dist = min([self.get_maze_distance(successor_pos, food) for food in food_list])
                    score -= min_food_dist * 100
                
                # If carrying food, consider going home
                if successor_state.num_carrying > 0:
                    if not successor_state.is_pacman:
                        score += 50000  # Bonus for scoring
                
                values.append(score)
            
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            return best_actions[0]
        
        # Check for invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        
        # If there are invaders, chase them aggressively
        if len(invaders) > 0:
            actions = game_state.get_legal_actions(self.index)
            values = []
            my_pos = my_state.get_position()
            
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_state = successor.get_agent_state(self.index)
                successor_pos = successor_state.get_position()
                
                # Find closest invader
                invader_distances = [self.get_maze_distance(successor_pos, inv.get_position()) for inv in invaders]
                min_invader_dist = min(invader_distances)
                
                # Prioritize catching invaders
                score = -min_invader_dist * 1000
                
                # Bonus for staying on defense
                if not successor_state.is_pacman:
                    score += 100
                
                # Penalty for stopping
                if action == Directions.STOP:
                    score -= 500
                
                values.append(score)
            
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            return best_actions[0]
        
        # No invaders - patrol around defense center and entry points
        else:
            actions = game_state.get_legal_actions(self.index)
            values = []
            my_pos = my_state.get_position()
            
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_state = successor.get_agent_state(self.index)
                successor_pos = successor_state.get_position()
                
                score = 0
                
                # Priority 1: Stay near the central boundary to intercept invaders early
                boundary_x = game_state.data.layout.width // 2
                if self.red:
                    boundary_x -= 1
                
                # Get middle y position for central patrol
                height = game_state.data.layout.height
                central_y = height // 2
                
                # Distance to central frontier position
                central_frontier_pos = (boundary_x, central_y)
                
                # Try to use maze distance, fallback to Manhattan if position invalid
                try:
                    distance_to_central_frontier = self.get_maze_distance(successor_pos, central_frontier_pos)
                except:
                    distance_to_central_frontier = abs(successor_pos[0] - boundary_x) + abs(successor_pos[1] - central_y)
                
                score -= distance_to_central_frontier * 100  # High priority
                
                # Priority 2: Stay near food clusters to protect them
                if len(self.food_clusters) > 0:
                    cluster_distances = [self.get_maze_distance(successor_pos, cluster) for cluster in self.food_clusters]
                    min_cluster_dist = min(cluster_distances)
                    score -= min_cluster_dist * 30  # Moderate priority
                
                # Priority 3: Maintain defensive position (stay as ghost)
                if not successor_state.is_pacman:
                    score += 500
                else:
                    # Heavy penalty for crossing into enemy territory
                    score -= 1000
                
                # Penalty for stopping - keep patrolling
                if action == Directions.STOP:
                    score -= 200
                
                values.append(score)
            
            max_value = max(values)
            best_actions = [a for a, v in zip(actions, values) if v == max_value]
            return best_actions[0]
    

