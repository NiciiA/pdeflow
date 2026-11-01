export function FieldDiagnostics({
  metrics,
  label = "Per-field reconstruction errors",
}: {
  metrics: Record<string, unknown>;
  label?: string;
}) {
  const fields = metrics.fields as
    | Record<string, { relative_l2_unobserved: number | null; mae: number }>
    | undefined;
  if (!fields)
    return (
      <p>Reconstruction errors unavailable without an evaluation reference.</p>
    );
  return (
    <div className="observation-table">
      <table aria-label={label}>
        <thead>
          <tr>
            <th>Physical field</th>
            <th>Unobserved relative L2</th>
            <th>MAE</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(fields).map(([name, value]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>
                {value.relative_l2_unobserved?.toPrecision(4) ?? "Unavailable"}
              </td>
              <td>{value.mae.toPrecision(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
