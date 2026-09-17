import json

from creator_search import CreatorDocument, CreatorSearch, SearchRequest


def main() -> None:
    documents = [
        CreatorDocument("delivery", "studio-7", "Digital asset delivery", "Signed download links expire after delivery.", True),
        CreatorDocument("update", "studio-7", "Subscriber update", "A weekly note for active subscribers.", True),
        CreatorDocument("public", "studio-7", "Processing notes", "How uploads are normalized before publishing."),
    ]
    search = CreatorSearch(documents)
    search.index()
    result = search.search(SearchRequest("studio-7", "asset delivery", subscriber=True))
    print(json.dumps([doc.__dict__ for doc in result], indent=2))


if __name__ == "__main__":
    main()

