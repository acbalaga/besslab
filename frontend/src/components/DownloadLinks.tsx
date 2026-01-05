import { Button } from "./ui/Button";

interface DownloadLinkProps {
  label: string;
  data: object | string;
  fileName: string;
}

export function DownloadLink({ label, data, fileName }: DownloadLinkProps) {
  const handleClick = () => {
    const payload = typeof data === "string" ? data : JSON.stringify(data, null, 2);
    const blob = new Blob([payload], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return <Button variant="secondary" size="sm" onClick={handleClick}>{label}</Button>;
}
