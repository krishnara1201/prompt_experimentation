interface ArmPairFieldsProps {
  arms: string[];
  labelA: string;
  labelB: string;
  valueA: string;
  valueB: string;
  onChangeA: (value: string) => void;
  onChangeB: (value: string) => void;
}

/** Two <select> arm pickers sharing the run's arm list, used by the
 *  equivalence and power panels. */
export function ArmPairFields({
  arms,
  labelA,
  labelB,
  valueA,
  valueB,
  onChangeA,
  onChangeB,
}: ArmPairFieldsProps) {
  return (
    <>
      {(
        [
          [labelA, valueA, onChangeA],
          [labelB, valueB, onChangeB],
        ] as const
      ).map(([label, value, onChange]) => (
        <label key={label} className="mb-2 block">
          <span className="text-xs uppercase text-gray-500">{label}</span>
          <select
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="mt-1 w-full rounded border px-2 py-1"
          >
            <option value="">Select an arm…</option>
            {arms.map((arm) => (
              <option key={arm} value={arm}>
                {arm}
              </option>
            ))}
          </select>
        </label>
      ))}
    </>
  );
}
