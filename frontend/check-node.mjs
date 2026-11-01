// Matches the intersection of Vite, Vitest and ESLint's installed engines.
import { pathToFileURL } from "node:url";
export function supported(version) {
  const [major, minor] = version.replace(/^v/, "").split(".").map(Number);
  return (major === 22 && minor >= 13) || major === 24 || major >= 26;
}
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href &&
  !supported(process.version)
) {
  console.error(
    `Unsupported Node ${process.version}. PDEFlow needs ^22.13.0 || ^24.0.0 || >=26.0.0 for its frontend tools. Run nvm install using the repository .nvmrc (22.23.3), then pdeflow setup. Headless Python training does not need Node.`,
  );
  process.exit(1);
}
