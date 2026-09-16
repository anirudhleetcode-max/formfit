import { EXERCISE_LIST, type ExerciseId } from "../pose/rules";

export default function ExercisePicker({ value, onChange, disabled }: {
  value: ExerciseId; onChange: (e: ExerciseId) => void; disabled?: boolean;
}) {
  return (
    <div className="seg" role="radiogroup" aria-label="Exercise">
      {EXERCISE_LIST.map((d) => (
        <button
          key={d.id}
          type="button"
          role="radio"
          aria-checked={value === d.id}
          className={"seg-btn" + (value === d.id ? " on" : "")}
          onClick={() => onChange(d.id)}
          disabled={disabled && value !== d.id}
          data-testid={`ex-${d.id}`}
        >
          {d.label}
        </button>
      ))}
    </div>
  );
}
