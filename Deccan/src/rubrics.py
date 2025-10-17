"""
Evaluation rubrics for assessing LLM responses
"""
from typing import Dict, Any
from dataclasses import dataclass

from .config import RUBRIC_DIMENSIONS


@dataclass
class RubricScore:
    """Represents a score for one dimension"""
    dimension: str
    score: int
    justification: str = ""


class EvaluationRubrics:
    """
    Define and manage evaluation rubrics for LLM responses
    """
    
    def __init__(self):
        self.dimensions = RUBRIC_DIMENSIONS
    
    def get_accuracy_guidelines(self) -> Dict[int, str]:
        """
        Accuracy scoring guidelines (0-5)
        
        Returns:
            Dictionary mapping scores to descriptions
        """
        return {
            0: "Completely incorrect or irrelevant information",
            1: "Mostly incorrect with minimal accurate content",
            2: "Some correct information but significant errors",
            3: "Partially accurate with notable gaps or minor errors",
            4: "Mostly accurate with minor imperfections",
            5: "Completely accurate and factually correct"
        }
    
    def get_helpfulness_guidelines(self) -> Dict[int, str]:
        """
        Helpfulness scoring guidelines (0-5)
        
        Returns:
            Dictionary mapping scores to descriptions
        """
        return {
            0: "Not helpful at all, does not address the question",
            1: "Minimally helpful, barely addresses the question",
            2: "Somewhat helpful but lacks important details",
            3: "Moderately helpful, provides useful but incomplete information",
            4: "Very helpful, addresses the question well with minor gaps",
            5: "Extremely helpful, comprehensive and actionable response"
        }
    
    def get_tone_guidelines(self) -> Dict[int, str]:
        """
        Tone/Bias scoring guidelines (0-5)
        
        Returns:
            Dictionary mapping scores to descriptions
        """
        return {
            0: "Highly inappropriate tone or significant bias",
            1: "Noticeably inappropriate or biased",
            2: "Somewhat inappropriate tone or slight bias",
            3: "Generally appropriate with minor tone issues",
            4: "Professional and appropriate tone with no significant bias",
            5: "Perfect tone, completely unbiased and professional"
        }
    
    def get_all_guidelines(self) -> Dict[str, Dict[int, str]]:
        """
        Get all rubric guidelines
        
        Returns:
            Dictionary of all dimensions with their scoring guidelines
        """
        return {
            "accuracy": self.get_accuracy_guidelines(),
            "helpfulness": self.get_helpfulness_guidelines(),
            "tone": self.get_tone_guidelines()
        }
    
    def validate_score(self, dimension: str, score: int) -> bool:
        """
        Validate if a score is within acceptable range
        
        Args:
            dimension: Name of the dimension
            score: Score value
            
        Returns:
            True if valid, False otherwise
        """
        if dimension not in self.dimensions:
            return False
        
        dim_config = self.dimensions[dimension]
        return dim_config["min_score"] <= score <= dim_config["max_score"]
    
    def print_rubrics(self):
        """Print all rubric guidelines in a readable format"""
        print("=" * 80)
        print("EVALUATION RUBRICS FOR LLM RESPONSES")
        print("=" * 80)
        
        all_guidelines = self.get_all_guidelines()
        
        for dimension, guidelines in all_guidelines.items():
            print(f"\n📊 {dimension.upper()}")
            print(f"   {self.dimensions[dimension]['description']}")
            print("-" * 80)
            
            for score, description in guidelines.items():
                print(f"   [{score}] {description}")
        
        print("\n" + "=" * 80)
        print("SCORING INSTRUCTIONS:")
        print("- Read the question and model response carefully")
        print("- Evaluate each dimension independently")
        print("- Provide justification for scores in edge cases")
        print("- Be consistent across all evaluations")
        print("=" * 80)


def main():
    """Example usage of EvaluationRubrics"""
    rubrics = EvaluationRubrics()
    rubrics.print_rubrics()
    
    # Example validation
    print("\nExample Validations:")
    print(f"accuracy=5: {rubrics.validate_score('accuracy', 5)}")
    print(f"accuracy=6: {rubrics.validate_score('accuracy', 6)}")
    print(f"invalid_dimension=3: {rubrics.validate_score('invalid_dimension', 3)}")


if __name__ == "__main__":
    main()
