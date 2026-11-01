import { Button } from "./components/ui/button";
import { requestItems } from "./science";
import type { MathematicalRequest, ParseReport } from "./mathematics";
export function RequestSummary({
  value,
  report,
  reviewed,
  onEdit,
}: {
  value: MathematicalRequest;
  report: ParseReport | null;
  reviewed: boolean;
  onEdit: (section: string, target: string) => void;
}) {
  return (
    <section className="request-summary" aria-label="Full problem summary">
      <div className="summary-heading">
        <span>YOUR PROBLEM</span>
        <span>𝒞 = (𝓔, λ, Ω, ℬ, ℐ, y, M)</span>
      </div>
      <div className="request-map">
        {requestItems(value, report, reviewed).map((item) => (
          <button
            key={item.symbol}
            onClick={() => onEdit(item.section, item.target)}
            aria-label={`Edit ${item.name}`}
          >
            <b>{item.symbol}</b>
            <span>
              <strong>{item.name}</strong>
              <small>{item.text}</small>
              <em
                data-warning={/Missing|unsupported|Incompatible|Review|Unsupported/.test(
                  item.status,
                )}
              >
                {item.status}
              </em>
            </span>
          </button>
        ))}
      </div>
      <p className="targets">
        <strong>What will be reconstructed?</strong>{" "}
        {value.declarations
          .filter(
            (d) =>
              d.status === "unknown" &&
              (d.kind === "field" || d.kind === "coefficient"),
          )
          .map((d) => `${d.name} (${d.meaning || "unknown field"})`)
          .join(" · ") || "Declare the unknown fields in your equation."}{" "}
        <Button
          size="sm"
          variant="link"
          onClick={() => onEdit("Problem", "physical-information")}
        >
          Edit targets
        </Button>
      </p>
    </section>
  );
}
