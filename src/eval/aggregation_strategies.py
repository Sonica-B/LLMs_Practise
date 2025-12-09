# aggregation_strategies.py
import numpy as np
from collections import Counter
from scipy.optimize import minimize

class AggregationStrategies:
    """Implements aggregation strategies for combining multiple responses."""
    
    @staticmethod
    def consistency_aggregation(esi_levels):
        """Measure agreement among ESI assessments."""
        if not esi_levels:
            return None, 0.0
        
        # Find most common ESI level
        esi_counter = Counter(esi_levels)
        most_common_esi, count = esi_counter.most_common(1)[0]
        
        # Calculate consistency
        consistency = count / len(esi_levels)
        
        return most_common_esi, consistency * 100  # Convert to percentage
    
    @staticmethod
    def avg_conf_aggregation(esi_levels, confidences):
        """Average confidences weighted by ESI level matches."""
        if not esi_levels or not confidences or len(esi_levels) != len(confidences):
            return None, 0.0
        
        # Find most common ESI level
        esi_counter = Counter(esi_levels)
        most_common_esi, _ = esi_counter.most_common(1)[0]
        
        # Sum confidences for the most common ESI level
        matching_confidences = [conf for esi, conf in zip(esi_levels, confidences) 
                              if esi == most_common_esi]
        
        # Calculate average confidence
        if matching_confidences:
            avg_confidence = sum(matching_confidences) / len(matching_confidences)
        else:
            avg_confidence = 0.0
        
        return most_common_esi, avg_confidence
    
    @staticmethod
    def pair_rank_aggregation(ranked_esi_levels):
        """Use ranking to estimate ESI level probabilities."""
        if not ranked_esi_levels:
            return None, 0.0
        
        # Extract all unique ESI levels
        all_esi_levels = set()
        for esi_list, _ in ranked_esi_levels:
            all_esi_levels.update(esi_list)
        
        # Count total occurrences and first-place occurrences
        esi_counts = {esi: 0 for esi in all_esi_levels}
        first_place_counts = {esi: 0 for esi in all_esi_levels}
        
        for esi_list, _ in ranked_esi_levels:
            if esi_list:
                # Count first place
                first_place_counts[esi_list[0]] += 1
                
                # Count all occurrences
                for esi in esi_list:
                    esi_counts[esi] += 1
        
        # Compute score based on both metrics
        esi_scores = {}
        for esi in all_esi_levels:
            # Weight first place more heavily
            score = (first_place_counts[esi] * 2 + esi_counts[esi]) / (2 * len(ranked_esi_levels) + len(ranked_esi_levels) * len(all_esi_levels))
            esi_scores[esi] = score * 100  # Convert to percentage
        
        # Find ESI with highest score
        best_esi = max(esi_scores.items(), key=lambda x: x[1])
        
        return best_esi[0], best_esi[1]